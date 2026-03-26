export INFERENCE_RESULTS_REPO_OWNER=${INFERENCE_RESULTS_REPO_OWNER:-mlcommons}
export INFERENCE_RESULTS_REPO_NAME=${INFERENCE_RESULTS_REPO_NAME:-mlperf_inference_submissions}
export INFERENCE_RESULTS_REPO_BRANCH=${INFERENCE_RESULTS_REPO_BRANCH:-main}
export INFERENCE_RESULTS_VERSION=${INFERENCE_RESULTS_VERSION:-v6.0}

if [ ! -e docs ]; then
    git clone https://github.com/mlcommons/inference_results_visualization_template.git docs
    test $? -eq 0 || exit $?
fi

cp docs/docinit.sh .
export default_division="closed";
export default_category="datacenter";

bash docinit.sh
