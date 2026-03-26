import asyncio
import queue
import time
import json
import requests
from argparse import ArgumentParser
from typing import List
from sglang.utils import wait_for_server
import multiprocessing as mp
import threading
import logging
import array
import numpy as np

from utils import RunnerArgs
from sglang.srt.server_args import ServerArgs

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("SUT_SGLang-8B")

import mlperf_loadgen as lg
from dataset import Dataset

class Instance(threading.Thread):
    def __init__(self, instance_id, 
                 url: str, 
                 model_path: str, 
                 dataset_path: str, 
                 input_queue=None, 
                 first_token_queue=None,
                 output_queue=None, 
                 cond_var=None, 
                 instances_ready=None, 
                 warmup: bool = False, 
                 streaming: bool = False,
                 async_server: bool = False,
                 counter=None,
                 output_log_dir: str = "output-logs"):
        """
        Initializes the Instance with the server URL and batch size.
        Args:
            instance_id (int): Unique identifier for the instance.
            url (str): The URL of the SGLang server.
            model_path (str): The path to the model.
            dataset_path (str): The path to the dataset.
            input_queue (multiprocessing.Queue): Queue for input requests. Items in input_queue are tuples, with mlperf data index (for dataset), and their corresponding query id.
            first_token_queue (multiprocessing.Queue): Queue for first token responses. Responses will be a tuple (id, first_token).
            output_queue (multiprocessing.Queue): Queue for output responses. Responses will be a tuple (id, output_ids).
            cond_var (threading.Condition): Condition variable for synchronizing access to shared resources.
            instances_ready (multiprocessing.Event): Event to signal when the instance is ready.
            warmup (bool): Flag indicating whether to perform warmup.
            streaming (bool): Flag indicating whether to enable streaming mode (Required for first token responses).
            async_server (bool): Flag indicating whether to use an async server (For server scenario).
            output_log_dir (str): Directory to save output log.
        Note:
            The `input_queue` should be populated with tuples of (rids, qindexes, tic) where `rids` is a list of request IDs and `qindexes` is a list of dataset indexes.
            The `output_queue` will receive tuples of (id, output_ids) where `id` corresponds to the request ID and `output_ids` is the
            generated output from the model.
        """
        super().__init__()
        self.instance_id = instance_id
        self.url = url
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.input_queue = input_queue
        self.first_token_queue = first_token_queue
        self.output_queue = output_queue
        self.cond_var = cond_var
        self.instances_ready = instances_ready
        self.warmup = warmup
        self.streaming = streaming
        self.use_async_server = async_server
        self.counter=counter
        self.output_log_dir = output_log_dir

        if self.streaming:
            self.run_inference = self.run_inference_stream
        else:
            self.run_inference = self.run_inference_lazy

        if self.use_async_server:
            self.async_tasks = set()
        else:
            self.async_tasks = None

        self.sampling_params = {"temperature": 0., "top_p": 1, "top_k": 1, "max_new_tokens": 128}

        self.tics = {}
        self.wait = {}

    def __del__(self):
        """
        Cleanup method to close any resources if needed.
        """
        if hasattr(self, 'fid'):
            self.fid.close()
            log.info(f"Closed file descriptor for instance {self.instance_id}.")

    def do_warmup(self):
        """
        Perform a warmup inference to ensure the model is loaded and ready.
        This method can be called before starting the main processing loop.
        """
        log.info(f"Warmup for instance {self.instance_id} with server at {self.url}")

        # Perform a dummy inference to warm up the model
        with open(f"prompt_llama4.json", "r") as f:
            dummy_prompts = json.load(f)["1024"]
        dummy_prompt = list(dummy_prompts.values())[:8]
        token_ids = self.data_obj.tokenizer.batch_encode_plus(dummy_prompt).input_ids
        json_data = {
            "input_ids": token_ids,
            "sampling_params": self.sampling_params,
            #"rid": ["warmup"]
        }
        log.info(f"Sending warmup request to {self.url} for instance {self.instance_id}")
        response = requests.post(
            f"{self.url}/generate",
            json=json_data,
        )
        if response.status_code != 200:
            log.error(f"Warmup request failed for instance {self.instance_id}: {response.text}")
            return
        log.info(f"Warmup request successful for instance {self.instance_id}.")

    def get_input_lens(self):
        """
        Returns the input lengths of the dataset.
        """
        if hasattr(self, 'data_obj'):
            return self.data_obj.getInputLengths()
        else:
            log.error("Dataset not loaded yet. Cannot get input lengths.")
            return []
        
    def run(self):
        # Load the model and dataset
        log.info(f"Loading dataset for instance {self.instance_id} from {self.dataset_path}")
        self.data_obj = Dataset(
            dataset_path=self.dataset_path,
            model_name=self.model_path,
        )
        
        self.data_obj.loadDataset()
        log.info(f"Dataset loaded for instance {self.instance_id}.")

        # Wait for the server to be ready
        log.info(f"Waiting for server at {self.url} to be ready...")
        # This will block until the server is ready to accept requests
        wait_for_server(self.url, timeout=60)

        # Perform warmup inference
        if self.warmup:
            log.info(f"Performing warmup for instance {self.instance_id}...")
            self.do_warmup()
            log.info(f"Warmup completed for instance {self.instance_id}.")
        
        # Signal that this instance is ready
        with self.instances_ready.get_lock():
            self.instances_ready.value += 1
            log.info(f"Instance {self.instance_id} is ready. Total instances ready: {self.instances_ready.value}")
        # Notify the condition variable that this instance is ready
        with self.cond_var:
            self.cond_var.notify_all()

        if self.use_async_server:
            asyncio.run(self.process_queries_async())
            # Wait for all async tasks to complete before exiting
            while self.async_tasks:
                time.sleep(0.1)
            log.info(f"All async tasks for instance {self.instance_id} have completed.")
        else:
            self.process_queries()

    async def process_queries_async(self):

        while True:

            qids, qindexes, tic = self.input_queue.get()
            if qids is None:
                log.info(f"Received termination signal. Exiting instance {self.instance_id}.")
                break
            
            token_ids,_,_,_ = self.data_obj.getSamples(qindexes)
            t = time.time()
            # print(f"Query fetched. ids: {qids}; len {[self.data_obj[q][1] for q in qindexes]}; tic: {tic}")
            task = asyncio.create_task(asyncio.to_thread(self.run_inference, token_ids, qids, tic, [t - i for i in tic]))
            await asyncio.sleep(.001)
            self.async_tasks.add(task)
            task.add_done_callback(self.async_tasks.discard)
        
    def process_queries(self):

        while True:
            qids, qindexes, tic = self.input_queue.get()
            if qids is None:
                print(f"Received termination signal. Exiting instance {self.instance_id}.")
                break
            token_ids,_,_,_ = self.data_obj.getSamples(qindexes)
            wait_time = time.time() - tic
            self.run_inference(token_ids, rids=qids, tic=tic, wait_time=wait_time)
        
    def run_inference_stream(self, token_ids, rids=List[str], tic=None, wait_time=None):
        json_data = {
            "input_ids": token_ids,
            "sampling_params": self.sampling_params,
            "rid": rids[0] if len(rids)==1 else rids, # Engine complains if token_ids is one prompt but rids is a list
            "stream": True
        }
        
        for i in range(len(rids)):
            self.tics[int(rids[i])] = tic[i]
            self.wait[int(rids[i])] = wait_time[i]
        
        response = requests.post(
            f"{self.url}/generate",
            json=json_data,
            stream=True
        )

        if response.status_code != 200:
            print(f"Error: {response.text}")
            return
        
        seen_ids = set()
        for chunk in response.iter_lines(decode_unicode=False):
            chunk = chunk.decode("utf-8")

            if chunk and chunk.startswith("data:"):
                if chunk == "data: [DONE]":
                    
                    continue
                data = json.loads(chunk[5:].strip("\n"))
                meta_info = data["meta_info"]
                id = int(meta_info['id'])
                output = data["output_ids"]
                proc_tic = time.time()
                if meta_info['finish_reason'] is not None:
                    self.output_queue.put((id, output, self.tics[id]))
                    # break

                if id in seen_ids:
                    continue
                
                # Add "First token" to the first_token_queue
                if self.first_token_queue is not None and len(output) > 0:
                    first_token = output[0]
                    self.first_token_queue.put((id, [first_token], self.tics[id], self.wait[id], meta_info['prompt_tokens']))
                    # with self.cond_var:
                    #     self.counter.value -= 1
                    #     self.cond_var.notify()
                    if time.time() - self.tics[id] >= 1.99:
                        print(f"{self.instance_id};{id} tot: {time.time() - self.tics[id]}s; ttft: {time.time() - self.tics[id] - self.wait[id]:.2f}s; wait: {self.wait[id]:.2f}s", flush=True)

                seen_ids.add(id)

    def run_inference_lazy(self, token_ids, rids=List[str], tic=None, wait_time=None):
        
        json_data = {
            "input_ids": token_ids,
            "sampling_params": self.sampling_params,
            "rid": rids
        }

        response = requests.post(
            f"{self.url}/generate",
            json=json_data,
        )

        if response.status_code != 200:
            print(f"Error: {response.text}")
            return
        
        outputs = response.json()
        for j, output in enumerate(outputs):
            id = int(output['meta_info']['id'])
            self.output_queue.put((id, output['output_ids'], tic))

class SUT():
    def __init__(self, runner_args: RunnerArgs=None, server_args: ServerArgs=None):
        self.model_path = server_args.model_path
        self.batch_size = runner_args.batch_size
        self.dataset_path = runner_args.dataset_path
        self.total_sample_count = runner_args.total_sample_count
        self.num_instances = runner_args.num_workers
        self.warmup = runner_args.warmup
        self.url = runner_args.url

        self.server_args = server_args
        self.runner_args = runner_args
        
        self.output_queue = mp.Queue()

        # Create a condition variable for synchronizing access to shared resources
        self.cond_var = mp.Condition()
        # Create multiprocessing.Value to accumulate the number of instances ready
        self.instances_ready = mp.Value('i', 0)

        # Non-streaming by default
        self.streaming = False
        self.first_token_queue = None
        self.use_async_server = False
        self.qsl = lg.ConstructQSL(self.total_sample_count, self.total_sample_count,
                                   self.LoadSamplesToRam, self.UnloadSamplesFromRam)
        self.current_counter_list = [mp.Value("i", 0) for _ in range(self.num_instances)]

    def start(self, serve_queue = False):
        # Create and start  Instances
        # Start multiple threads for issuing requests            
        
        self.instances = []
        self.counter = 0
        self.input_queue =  [mp.Queue() for _ in range(self.num_instances)] if serve_queue else mp.Queue()
        for i in range(self.num_instances):
            # Create mulitprocessing.Event to signal when instances are ready
            iq = self.input_queue[i] if serve_queue else self.input_queue
            instance = Instance(
                instance_id=i,
                model_path=self.model_path,
                dataset_path=self.dataset_path,
                url=self.url[i % len(self.url)],
                input_queue=iq,
                first_token_queue=self.first_token_queue,
                output_queue=self.output_queue,
                cond_var=self.cond_var,
                instances_ready=self.instances_ready,
                warmup=self.warmup,
                streaming=self.streaming,
                output_log_dir=self.runner_args.output_log_dir,
                async_server=self.use_async_server,
                counter=self.current_counter_list[i]
            )
            instance.start()
            self.instances.append(instance)

        # Start a thread to process outputs
        self.response_thread = threading.Thread(target=self.response_loadgen, daemon=True)

        # Wait for all instances to be ready
        print("Waiting for all instances to be ready.")
        with self.cond_var:
            self.cond_var.wait_for(lambda: self.instances_ready.value == self.num_instances)

        # Get the input lengths from the first instance
        if self.instances:
            self.input_lengths = self.instances[0].get_input_lens()
           
        self.response_thread.start()

    def get_best_rank(self, value_added):
        current_counters = np.array([(self.current_counter_list[i].value+value_added) for i in range(self.num_instances)])
        target_rank = np.argmin(current_counters)
        return target_rank

    def issue_queries(self, query_samples):
        # If the size of query_samples is more than 1, that's Offline scenario, so we need to process them in batches
        # First sort the query samples indexes by dataset input lengths
        tic = time.time()
        query_samples.sort(key=lambda x: -self.input_lengths[x.index])
        j = 0
        for start_index in range(0, len(query_samples), self.batch_size):
            end_index = min(start_index + self.batch_size, len(query_samples))
            qindexes = [query_samples[i].index for i in range(start_index, end_index)]
            qids = [str(query_samples[i].id) for i in range(start_index, end_index)]

            # Put the batch into the input queue
            # with self.cond_var:
            #     target_rank = self.get_best_rank(len(qids))
            #     self.current_counter_list[target_rank].value += len(qids)    
            self.input_queue.put((qids, qindexes, tic))

    def stop(self, serve_queue = False):
        # Stop all instances by sending a termination signal
        print("Stopping all instances.")
        for i in range(self.num_instances):
            if serve_queue:
                self.input_queue[i].put((None, None, None))
            else:
                self.input_queue.put((None, None, None))  # Send termination signal to each instance

        # Wait for all threads to finish
        for instance in self.instances:
            instance.join()

        # Signal the output queue that processing is done
        self.output_queue.put((None, None, None))
        self.response_thread.join()

    def response_loadgen(self):
        """
        This method runs in a separate thread to continuously collect outputs from the output queue.
        It sends outputs to loadgen
        """
        num_processed = 0
        timer = time.time()
        while True:
            qid, processed_output, tic = self.output_queue.get()
            if qid is None:
                break  # Exit condition for the thread
            n_tokens = len(processed_output)
            response_array = array.array("B", np.array(processed_output, np.int32).tobytes())
            bi = response_array.buffer_info()
            response = [lg.QuerySampleResponse(qid, bi[0], bi[1], n_tokens)]
            lg.QuerySamplesComplete(response)
            num_processed += 1
            # Add progress bar tracking num_processed per second
            if num_processed % 250 == 0 or num_processed == 13368:
                elapsed_time = time.time() - timer
                if elapsed_time > 0:
                    print(f"Rate: {num_processed / elapsed_time:.2f} queries/sec; Processed: {num_processed} queries", flush=True)
                    #timer = time.time()

    def flush_queries(self):
        pass

    def __del__(self):
        pass

    def get_qsl(self):
        return self.qsl
    
    def LoadSamplesToRam(self, query_samples):
        pass

    def UnloadSamplesFromRam(self, query_samples):
        pass


class SUTServer(SUT):
    def __init__(self, runner_args: RunnerArgs=None, server_args: ServerArgs=None):
        super().__init__(runner_args, server_args)
        self.first_token_queue = mp.Queue()
        self.query_queue = mp.Queue()
        self.streaming = True  # Enable streaming mode for first token responses
        self.use_async_server = runner_args.use_async_server
        self.finished = False

        self.temp = 0
        
    def start(self):
        # Create first token response thread
        print(f"Starting first-token response thread")
        self.ft_response_thread = threading.Thread(target=self.process_first_tokens)
        self.ft_response_thread.daemon = True
        self.ft_response_thread.start()

        if self.runner_args.batch_size > 1:
            print(f"Starting query batcher thread")
            self.batcher = threading.Thread(target=self.query_batcher, daemon=True)
            self.batcher.start()

        # Start the main SUT server
        super().start(True)

    def query_batcher(self):
        """Batch the arriving queries from loadgen"""
        ids = []
        indexes = []
        tics = []
        input_len_list = []
        time_compute_list = []
        time_start_lists = []
        flag_lists = []
        time_left = 1.8
        time_limit_fin = 0.35
        time_limit_add = 0.25
        j = 0
        while True:
            new_query = False
            try:
                qid, index, rec_time = self.query_queue.get(False)

                if qid is None:
                    for i in range(self.num_instances):
                        self.input_queue[i].put((None, None, None))
                    break
            except: # queue.Empty:
                # If the first item in the queue has waited for too long, send the current batch
                pass
            else:
                if qid is None:
                    for i in range(self.num_instances):
                        self.input_queue[i].put((None, None, None))
                    break
                
                input_len = self.input_lengths[index]

                if input_len > 1600:
                    val = 0.000575
                elif input_len > 1000:
                    val = 0.000625
                # elif input_len > 500:
                #     val = 0.00065
                # else:
                #     val = 0.00075
                else:
                    val = 0.000675
                
                time_compute_c = val * input_len
                start_time_c = time.time()
                
                if time_compute_c > time_left:
                        # with self.cond_var:
                        #     target_rank = self.get_best_rank(1)
                        #     self.current_counter_list[target_rank].value += 1
                        #     self.cond_var.notify()
                        self.input_queue[j].put(([qid], [index], [rec_time]))
                        j += 1
                        if j % self.num_instances == 0:
                            j = 0
                else:
                    new_query = True
                
            if new_query:
                if len(ids)==0 or input_len > 1900:
                    ids.append([qid])
                    indexes.append([index])
                    tics.append([rec_time])
                    time_compute_list.append(time_compute_c)
                    time_start_lists.append([start_time_c])
                    input_len_list.append(input_len)
                    
                    if input_len > 1600:
                        flag_lists.append(2)
                    elif input_len > 1000:
                        flag_lists.append(1)
                    # elif input_len > 500:
                    #     flag_lists.append(1)
                    else:
                        flag_lists.append(0)
                else:
                    for i, (id, ind, tic) in enumerate(zip(ids, indexes, tics)):
                        if (1600 >= input_len > 1000 and flag_lists[i] == 1) or (input_len <= 1000 and flag_lists[i] == 0) : # (1000 >= input_len > 500 and flag_lists[i] == 1) or (input_len <= 500 and flag_lists[i] == 0):                        
                            time_compute = time_compute_list[i]
                            time_wait = np.max(time.time()-np.array(time_start_lists[i]))

                            if flag_lists[i] == 1:
                                val = 0.000675
                            elif flag_lists[i] == 0:
                                if len(id) in [1,2]:
                                    val = 0.000625
                                else:
                                    val = 0.000575

                            time_compute = (input_len + input_len_list[i]) * val
                            time_needed = time_compute + time_wait
                            if time_needed<time_left-time_limit_add:
                                id.append(qid)
                                ind.append(index)
                                tic.append(rec_time)
                                time_compute_list[i] += time_compute_c
                                time_start_lists[i].append(start_time_c)
                                input_len_list[i] += input_len
                                break
                        # Cannot insert into existing lists
                        if i==len(ids)-1:
                            ids.append([qid])
                            indexes.append([index])
                            tics.append([rec_time])
                            time_compute_list.append(time_compute_c)
                            time_start_lists.append([start_time_c])
                            input_len_list.append(input_len)

                            if input_len > 1600:
                                flag_lists.append(2)
                            elif input_len > 1000:
                                flag_lists.append(1)
                            # elif input_len > 500:
                            #     flag_lists.append(1)
                            # else:
                            #     flag_lists.append(0)
                            else:
                                flag_lists.append(0)
                            
                            break

            for i in range(len(ids)-1, -1, -1):
                id = ids[i]
                ind = indexes[i]
                tic = tics[i]
                time_wait = np.max(time.time()-np.array(time_start_lists[i]))
                time_needed = time_compute_list[i] + time_wait
                if time_needed>time_left-time_limit_fin:
                    # with self.cond_var:
                    #     target_rank = self.get_best_rank(len(id))
                    #     self.current_counter_list[target_rank].value += len(id)
                    #     self.cond_var.notify()
                    
                    self.input_queue[j].put((id, ind, tic))
                    j += 1
                    if j % self.num_instances == 0:
                        j = 0
                    # print(f"qid {id} to rank {target_rank}; tc {time_compute_list[i]}; len {input_len_list[i]}")
                    del ids[i]
                    del indexes[i]
                    del tics[i]
                    del time_compute_list[i]
                    del time_start_lists[i]
                    del flag_lists[i]
                    del input_len_list[i]

    def issue_queries(self, query_samples):
        if self.runner_args.batch_size > 1:
            for q in query_samples:
                self.query_queue.put((str(q.id), q.index, time.time()))
        else:
            # with self.cond_var:
            #     target_rank = self.get_best_rank(len(id))
            #     self.current_counter_list[target_rank].value += len(id)
            #     self.cond_var.notify()
            self.input_queue[self.temp].put(([str(query_samples[0].id)], [query_samples[0].index], [time.time()]))
            self.temp += 1
            if self.temp % self.num_instances == 0:
                self.temp = 0
            
    def process_first_tokens(self):
        while True:
            qid, processed_output, tic, wait_time, inp_len = self.first_token_queue.get()
            if qid is None:
                print("Exiting First token response thread")
                break
            if time.time() - tic > 1.975:
                self.counter += 1
                print(f"ttft {time.time() - tic}; violation count: {self.counter}")
            
            response_data = array.array("B", np.array(processed_output, np.int32).tobytes())
            buf = response_data.buffer_info()
            response = [lg.QuerySampleResponse(qid, buf[0], buf[1])]
            lg.FirstTokenComplete(response)

    def stop(self):
        # Stop the first token response thread
        print("Stopping first token response thread.")
        self.first_token_queue.put((None, None, None, None, None))  # Send termination signal
        # Stop the query batcher thread if it exists
        print("Stopping query batcher thread.")
        self.query_queue.put((None, None, None))
        self.ft_response_thread.join()
        # Stop the main SUT server
        super().stop(True)
