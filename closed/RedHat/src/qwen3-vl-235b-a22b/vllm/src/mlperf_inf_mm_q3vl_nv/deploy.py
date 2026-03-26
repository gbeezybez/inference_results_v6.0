"""Endpoint deployers for deploying and managing the lifecycles of VLM endpoints."""

from __future__ import annotations

import os
import socket
import subprocess
import time
from collections.abc import Callable, Sequence
from datetime import timedelta
from functools import partial
from pathlib import Path
from urllib.parse import urlparse, urlunparse
from types import TracebackType
from typing import Self

import requests
from huggingface_hub import repo_exists, snapshot_download
from loguru import logger
from mlperf_inf_mm_q3vl.deploy import (
    EndpointDeployer,
    LocalProcessDeadError,
    LocalVllmDeployer,
)
from mlperf_inf_mm_q3vl.log import get_log_file_path
from mlperf_inf_mm_q3vl.schema import Settings
from mpi4py import MPI

from .schema import DynamoEndpoint, DynamoVllmLaunchConfig
from .wandb_utils import log_launch_config_to_wandb

HTTP_OK = 200


class ServiceHealthcheckTimeoutError(RuntimeError):
    """Raised when a service fails to be healthy within the specified timeout."""

    def __init__(self, timeout: timedelta) -> None:
        """Initialize the exception.

        Args:
            timeout: The timeout duration that was exceeded.
        """
        super().__init__(
            f"Service failed to be healthy within the timeout of {timeout}.",
        )


def wait_for_healthy(
    wait_timeout: timedelta,
    poll_interval: timedelta,
    failfast: Callable[[], None],
    healthcheck: Callable[[], bool],
) -> None:
    """Wait for a service to be healthy via healthcheck."""
    start_time = time.time()
    while time.time() - start_time < wait_timeout.total_seconds():
        failfast()
        logger.info(
            "Waiting {:0.2f} seconds for service to be healthy...",
            time.time() - start_time,
        )
        if healthcheck():
            logger.info("Service is now healthy!")
            return

        time.sleep(poll_interval.total_seconds())

    raise ServiceHealthcheckTimeoutError(wait_timeout)


def child_process_failfast(process: subprocess.Popen) -> None:
    """Failfast the child process."""
    returncode = process.poll()
    if returncode is not None:
        raise LocalProcessDeadError(
            returncode=returncode,
            stdout_file_path=process.stdout.name,
            stderr_file_path=process.stderr.name,
        )


def healthcheck_by_status_code(
    healthcheck_url: str,
    healthcheck_timeout: timedelta,
) -> bool:
    """Healthcheck the service by checking the status code."""
    try:
        logger.info("Healthchecking {}...", healthcheck_url)
        response = requests.get(
            healthcheck_url,
            timeout=healthcheck_timeout.total_seconds(),
        )
        if response.status_code == HTTP_OK:
            logger.info(
                "Service with health check URL {} is now healthy!",
                healthcheck_url,
            )
            return True
    except requests.exceptions.RequestException:
        return False


class DynamoFrontendStartupTimeoutError(RuntimeError):
    """Raised when the Dynamo frontend fails to start within the specified timeout."""

    def __init__(self, timeout: timedelta) -> None:
        """Initialize the exception.

        Args:
            timeout: The timeout duration that was exceeded.
        """
        super().__init__(
            f"Dynamo frontend failed to start within the timeout of {timeout}.",
        )


class DynamoFrontendFormatError(RuntimeError):
    """Raised when the Dynamo frontend returns backend information with unexpected format."""

    def __init__(self, response_text: str) -> None:
        """Initialize the exception.

        Args:
            response_exception: The response obtained from the Dynamo frontend.
        """
        super().__init__(
            f"Dynamo frontend returned backend information with unexpected format: {response_text}"
        )


class DynamoVllmProcessDeadError(RuntimeError):
    """Raised when the Dynamo vLLM process dies."""

    def __init__(self, ranks: Sequence[int]) -> None:
        """Initialize the exception."""
        super().__init__(f"Dynamo vLLM processes on ranks {ranks} have died.")


class UnclearModelRepoIdError(RuntimeError):
    """Raised when the model repo ID is unclear."""

    def __init__(self, repo_id: str) -> None:
        """Initialize the exception."""
        super().__init__(
            f"The model repo ID {repo_id} is unclear. It is neither a model repo ID on "
            "Hugging Face Hub, nor a local directory.",
        )


class MpiDynamoVllmEndpointDeployer(EndpointDeployer):
    """The endpoint deployer for the NVIDIA-optimized Qwen3-VL (Q3VL) benchmark."""

    def __init__(self, endpoint: DynamoEndpoint, settings: Settings) -> None:
        """Initialize the MPI-Dynamo-vLLM endpoint deployer.

        Args:
            endpoint: The configuration for the MPI-Dynamo-vLLM-based endpoint.
            settings: The settings for the benchmark.
        """
        super().__init__(endpoint=endpoint, settings=settings)
        self.endpoint = endpoint

        self.comm_world = MPI.COMM_WORLD
        self.comm_local = self.comm_world.Split_type(MPI.COMM_TYPE_SHARED)
        self.rank = self.comm_world.Get_rank()
        self.local_rank = self.comm_local.Get_rank()
        self.world_size = self.comm_world.Get_size()
        self.local_size = self.comm_local.Get_size()

        self._determine_endpoint_url()

        self._etcd_process: subprocess.Popen | None = None
        self._nats_process: subprocess.Popen | None = None
        self._dynamo_frontend_process: subprocess.Popen | None = None
        self._dynamo_vllm_process: subprocess.Popen | None = None

    def _determine_endpoint_url(self) -> str:
        """Determine the endpoint URL.

        Because which node to allocate is decided by the job scheduler, here we need to
        determine the hostname of the head node dynamically.
        """
        head_node_hostname = None
        if self.rank == 0:
            head_node_hostname = socket.gethostbyname(socket.gethostname())
        head_node_hostname = self.comm_world.bcast(head_node_hostname, root=0)
        parsed_url = urlparse(self.endpoint.url)
        # Preserve the port if it exists
        new_netloc = (
            f"{head_node_hostname}:{parsed_url.port}"
            if parsed_url.port
            else head_node_hostname
        )
        self.endpoint.url = urlunparse(
            parsed_url._replace(netloc=new_netloc),
        )
        self.endpoint.etcd.hostname = head_node_hostname
        self.endpoint.nats.hostname = head_node_hostname
        logger.info(
            "After head node discovery, the endpoint settings are: {}",
            self.endpoint,
        )

    def _log_file_path(self, key: str) -> Path:
        """Get the log file path for a given key.

        Args:
            key: The key for the log file.

        Returns:
            The log file path.
        """
        return get_log_file_path(key + f".rank{self.rank}", self.settings)

    def _launch_process(
        self,
        name: str,
        cmd: list[str],
        env: dict[str, str],
    ) -> subprocess.Popen:
        stdout_file_path = self._log_file_path(f"{name}.stdout")
        stderr_file_path = self._log_file_path(f"{name}.stderr")
        logger.info("Starting {} with command: {}", name, cmd)
        logger.info("Starting {} with environment variables: {}", name, env)
        process = subprocess.Popen(
            cmd,
            stdout=stdout_file_path.open("w"),
            stderr=stderr_file_path.open("w"),
            text=True,
            env=env,
        )
        logger.info("Started {} process with PID: {}", name, process.pid)
        logger.info("{} stdout will be logged to: {}", name, stdout_file_path)
        logger.info("{} stderr will be logged to: {}", name, stderr_file_path)
        return process

    def _launch_process_and_wait_for_healthy(
        self,
        name: str,
        cmd: list[str],
        env: dict[str, str],
        healthcheck_url: str,
        wait_timeout: timedelta,
        healthcheck_timeout: timedelta,
        poll_interval: timedelta,
    ) -> subprocess.Popen:
        process = self._launch_process(name=name, cmd=cmd, env=env)
        wait_for_healthy(
            wait_timeout=wait_timeout,
            poll_interval=poll_interval,
            failfast=partial(child_process_failfast, process),
            healthcheck=partial(
                healthcheck_by_status_code,
                healthcheck_url,
                healthcheck_timeout,
            ),
        )
        return process

    def _maybe_download_model(self) -> None:
        if repo_exists(
            repo_id=self.endpoint.model.repo_id,
            repo_type="model",
            token=self.endpoint.model.token,
        ):
            logger.info(
                "{} exists on Hugging Face Hub and will be attempted to be downloaded.",
                self.endpoint.model.repo_id,
            )
            snapshot_download(
                repo_id=self.endpoint.model.repo_id,
                repo_type="model",
                revision=self.endpoint.model.revision,
                token=self.endpoint.model.token,
            )
        elif Path(self.endpoint.model.repo_id).is_dir():
            logger.info(
                "{} is a local directory and will be used.",
                self.endpoint.model.repo_id,
            )
        else:
            raise UnclearModelRepoIdError(self.endpoint.model.repo_id)

    def _launch_etcd_and_wait_for_healthy(self) -> None:
        cmd = [
            "etcd",
            "--listen-client-urls",
            f"http://{self.endpoint.etcd.hostname}:{self.endpoint.etcd.port}",
            "--advertise-client-urls",
            f"http://{self.endpoint.etcd.hostname}:{self.endpoint.etcd.port}",
            "--data-dir",
            "/tmp/etcd",
        ]
        self._etcd_process = self._launch_process_and_wait_for_healthy(
            name="etcd",
            cmd=cmd,
            env=os.environ.copy(),
            healthcheck_url=f"http://{self.endpoint.etcd.hostname}:{self.endpoint.etcd.port}/health",
            wait_timeout=self.endpoint.etcd.startup_timeout,
            healthcheck_timeout=self.endpoint.etcd.healthcheck_timeout,
            poll_interval=self.endpoint.etcd.poll_interval,
        )

    def _launch_nats_and_wait_for_healthy(self) -> None:
        cmd = [
            "nats-server",
            "-js",
            "-a",
            self.endpoint.nats.hostname,
            "-p",
            str(self.endpoint.nats.port),
            "-m",
            str(self.endpoint.nats.monitoring_port),
        ]
        self._nats_process = self._launch_process_and_wait_for_healthy(
            name="nats",
            cmd=cmd,
            env=os.environ.copy(),
            healthcheck_url=f"http://{self.endpoint.nats.hostname}:{self.endpoint.nats.monitoring_port}/healthz",
            wait_timeout=self.endpoint.nats.startup_timeout,
            healthcheck_timeout=self.endpoint.nats.healthcheck_timeout,
            poll_interval=self.endpoint.nats.poll_interval,
        )

    def _launch_dynamo_frontend_and_wait_for_healthy(self) -> None:
        parsed_url = urlparse(self.endpoint.url)
        cmd = [
            "--http-host",
            parsed_url.hostname,
            "--http-port",
            str(parsed_url.port),
        ]
        cmd.extend(self.endpoint.frontend.cli)
        env = os.environ.copy()
        env_updates: dict[str, str] = {
            "ETCD_ENDPOINTS": f"http://{self.endpoint.etcd.hostname}:{self.endpoint.etcd.port}",
            "NATS_SERVER": f"nats://{self.endpoint.nats.hostname}:{self.endpoint.nats.port}",
        }
        # Pass HF_TOKEN so frontend can fetch model metadata for gated models
        if self.endpoint.model.token:
            env_updates["HF_TOKEN"] = self.endpoint.model.token
        env.update(env_updates)

        log_launch_config_to_wandb(
            name="dynamo.frontend",
            cmd=cmd,
            env=env,
        )
        
        self._dynamo_frontend_process = self._launch_process_and_wait_for_healthy(
            name="dynamo.frontend",
            cmd=["python3","-m","dynamo.frontend"] + cmd,
            env=env,
            healthcheck_url=urlunparse(parsed_url._replace(path="/health")),
            wait_timeout=self.endpoint.frontend.startup_timeout,
            healthcheck_timeout=self.endpoint.frontend.healthcheck_timeout,
            poll_interval=self.endpoint.frontend.poll_interval,
        )

    @staticmethod
    def get_num_gpus_from_vllm_endpoint(vllm: DynamoVllmLaunchConfig) -> int:
        """Calculate the number of GPUs required from VllmEndpoint CLI arguments.

        Parses --tensor-parallel-size (or -tp), --pipeline-parallel-size (or -pp),
        and --data-parallel-size (or -dp) from the CLI arguments and returns their product.

        Args:
            endpoint: The VllmEndpoint configuration.

        Returns:
            The total number of GPUs required (tensor_parallel * pipeline_parallel * data_parallel).
        """
        cli = vllm.cli

        # Refer: https://github.com/vllm-project/vllm/blob/83e1c76dbe07e30b7f4e6dbe17ba580f4afc98f0/vllm/model_executor/layers/fused_moe/config.py#L895
        parallel_param_names = ["tensor", "pipeline", "data", "prefill-context"]
        parallel_param_names = [param + "-parallel-size" for param in parallel_param_names]
        parallel_param_names.extend([param.replace("-", "_") for param in parallel_param_names])
        parallel_param_names = ["--" + param for param in parallel_param_names]
        parallel_param_names += ["-tp", "-pp", "-dp", "-pcp"]

        parallel_size = 1
        for idx, arg in enumerate(cli):
            if arg in parallel_param_names:
                if idx + 1 < len(cli):
                    parallel_size *= int(cli[idx + 1])
            elif "=" in arg:
                param, value = arg.split("=", 1)
                if param in parallel_param_names:
                    parallel_size *= int(value)

        return parallel_size

    def _launch_dynamo_vllm(self) -> None:
        cmd = [
            "--model",
            self.endpoint.model.repo_id,
            "--revision",
            self.endpoint.model.revision,
        ]
        if self.endpoint.model.token:
            cmd.extend(["--hf-token", self.endpoint.model.token])
        if self.endpoint.api_key:
            cmd.extend(["--api-key", self.endpoint.api_key])
        cmd.extend(self.endpoint.vllm.cli)
        env = os.environ.copy()
        num_gpus = self.get_num_gpus_from_vllm_endpoint(self.endpoint.vllm)
        env_updates: dict[str, str] = {
            "ETCD_ENDPOINTS": f"http://{self.endpoint.etcd.hostname}:{self.endpoint.etcd.port}",
            "NATS_SERVER": f"nats://{self.endpoint.nats.hostname}:{self.endpoint.nats.port}",
            "CUDA_VISIBLE_DEVICES": ",".join(
                str(i)
                for i in range(
                    self.local_rank * num_gpus,
                    (self.local_rank + 1) * num_gpus,
                )
            ),
        }
        # Also set HF_TOKEN env var for vLLM compatibility
        if self.endpoint.model.token:
            env_updates["HF_TOKEN"] = self.endpoint.model.token
        env.update(env_updates)

        log_launch_config_to_wandb(
            name="dynamo.vllm",
            cmd=cmd,
            env=env,
        )

        self._dynamo_vllm_process = self._launch_process(
            name="dynamo.vllm",
            cmd=["python3","-m","dynamo.vllm"] + cmd,
            env=env,
        )

    def _startup(self):
        if self.rank == 0:
            self._maybe_download_model()
            self._launch_etcd_and_wait_for_healthy()
            self._launch_nats_and_wait_for_healthy()
            self._launch_dynamo_frontend_and_wait_for_healthy()
        self.comm_world.Barrier()
        self._launch_dynamo_vllm()

    def _check_num_backends(self, healthcheck_timeout: timedelta) -> bool:
        """Check if the number of backends is correct"""

        response = requests.get(
            urlunparse(urlparse(self.endpoint.url)._replace(path="/health")),
            timeout=healthcheck_timeout.total_seconds()
        )
        response_json = response.json()

        try:
            instances = response_json["instances"]
            num_backends = len(
                [
                    instance
                    for instance in instances
                    if instance["endpoint"] == "generate"
                ],
            )
        except (TypeError, KeyError) as e:
            raise DynamoFrontendFormatError(response.text) from e

        if num_backends < self.world_size:
            logger.info(
                "Waiting for {} backends to be ready...",
                self.world_size - num_backends,
            )
            return False
        return True

    def _failfast(self) -> None:
        fails = self.comm_world.allgather(self._dynamo_vllm_process.poll() is not None)
        if any(fails):
            raise DynamoVllmProcessDeadError(
                ranks=[i for i, fail in enumerate(fails) if fail],
            )

    def _wait_for_ready(self):
        wait_for_healthy(
            wait_timeout=self.endpoint.startup_timeout,
            poll_interval=self.endpoint.poll_interval,
            failfast=self._failfast,
            healthcheck=partial(self._check_num_backends, self.endpoint.healthcheck_timeout)
        )

    def _terminate_or_kill_process(self, proc: subprocess.Popen):
        if proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=self.endpoint.shutdown_timeout.total_seconds())
            logger.info("Process terminated gracefully")
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            logger.info("Process killed")

    def _shutdown(self):
        for proc in [
            self._dynamo_vllm_process,
            self._dynamo_frontend_process,
            self._etcd_process,
            self._nats_process,
        ]:
            if proc is not None:
                self._terminate_or_kill_process(proc)



class VllmProfileEndpointDeployer(LocalVllmDeployer):
    def __enter__(self) -> Self:
        """Original enter function with profile start check"""
        super().__enter__()
        if self.endpoint.profile:
            self._start_profile()
        return self

    def __exit__(self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if self.endpoint.profile:
            self._stop_profile()
        super().__exit__(exc_type, exc_val, exc_tb)


    def _start_profile(self) -> None:
        profile_url = self.endpoint.url.rstrip("/v1") + "/start_profile"
        try:
            response = requests.post(
                profile_url,
                timeout=self.endpoint.payload_timeout.total_seconds(),
            )
            if response.status_code == HTTP_OK:
                logger.info("Profile started successfully")
                return
        except requests.exceptions.RequestException as e:
            logger.warning(f"Failed to start profile: {e}")

    def _stop_profile(self) -> None:
        profile_url = self.endpoint.url.rstrip("/v1") + "/stop_profile"
        try:
            response = requests.post(
                profile_url,
                timeout=self.endpoint.payload_timeout.total_seconds(),
            )
            if response.status_code == HTTP_OK:
                logger.info("Profile stopped successfully")
                return
        except requests.exceptions.RequestException:
            pass
