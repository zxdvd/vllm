"""The tests in this file verify end-to-end speculative decoding correctness.

This docstring details important information on the testing methodology.

Most of the tests rely on "greedy equality", where we expect the output of
speculative decoding on a sequence to exactly match the output of normal non-
speculative decoding.

Since speculative decoding with rejection sampling guarantees that the output
distribution matches the target model's output distribution (up to hardware
numerics, see https://arxiv.org/pdf/2302.01318.pdf), we can expect greedy
equality.

For ngram lookup, its idea comes from https://github.com/apoorvumang/prompt-lookup-decoding,
and is merged into transform code base: https://github.com/huggingface/transformers/pull/27775.
Since there is no model is needed for generate the proposal, we could make
the testcase much simplier than drafter multi-step one.

However, we still need to verify below scenario could be passed:
    * Batch size 1 greedy equality
    * Batch size >1 greedy equality
    * Test greedy equality under preemption
    * Test greedy equality under various ngram sizes / speculative sizes

With those tests, we can say at least, ngram spec would not break the correctess
for the target model outputs.
"""

from itertools import cycle

import pytest

from vllm import SamplingParams

from .conftest import get_output_from_llm_generator


@pytest.mark.parametrize(
    "common_llm_kwargs",
    [{
        # Skip cuda graph recording for fast test.
        "enforce_eager": True,

        # Required for spec decode.
        "use_v2_block_manager": True,

        # Print spec metrics.
        "disable_log_stats": False,
    }])
@pytest.mark.parametrize("per_test_common_llm_kwargs", [
    {
        "model": "JackFram/llama-68m",
    },
])
@pytest.mark.parametrize("baseline_llm_kwargs", [{}])
@pytest.mark.parametrize("test_llm_kwargs", [
    {
        "speculative_model": "[ngram]",
        "num_speculative_tokens": 5,
        "ngram_prompt_lookup_max": 3,
    },
])
@pytest.mark.parametrize(
    "output_len",
    [
        # Use long output len for the small model test.
        1536,
    ])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("seed", [1])
def test_spec_decode_e2e_greedy_correctness_tiny_model_bs1(
        baseline_llm_generator, test_llm_generator, batch_size: int,
        output_len: int):
    """Verify greedy equality on a tiny model with batch size of one.

    Since this test is cheaper than other e2e correctness tests, we generate
    with a higher output_len.
    """
    run_greedy_equality_correctness_test(baseline_llm_generator,
                                         test_llm_generator,
                                         batch_size,
                                         max_output_len=output_len,
                                         force_output_len=True)


@pytest.mark.parametrize(
    "common_llm_kwargs",
    [{
        # Skip cuda graph recording for fast test.
        "enforce_eager": True,

        # Required for spec decode.
        "use_v2_block_manager": True,

        # Print spec metrics.
        "disable_log_stats": False,
    }])
@pytest.mark.parametrize("per_test_common_llm_kwargs", [
    {
        "model": "JackFram/llama-68m",
    },
])
@pytest.mark.parametrize("baseline_llm_kwargs", [{}])
@pytest.mark.parametrize("test_llm_kwargs", [
    {
        "speculative_model": "[ngram]",
        "num_speculative_tokens": 5,
        "ngram_prompt_lookup_max": 3,
    },
])
@pytest.mark.parametrize(
    "output_len",
    [
        # Use small output len for fast test.
        256,
    ])
@pytest.mark.parametrize("batch_size", [64])
@pytest.mark.parametrize("seed", [1])
def test_spec_decode_e2e_greedy_correctness_tiny_model_large_bs(
        baseline_llm_generator, test_llm_generator, batch_size: int,
        output_len: int):
    """Verify greedy equality on a tiny model and large batch size.
    """
    run_greedy_equality_correctness_test(baseline_llm_generator,
                                         test_llm_generator,
                                         batch_size,
                                         max_output_len=output_len,
                                         force_output_len=True)


@pytest.mark.parametrize(
    "common_llm_kwargs",
    [{
        "block_size": 8,
        # 2 for small prompt, 256//8 for generated.
        "num_gpu_blocks_override": 2 + 256 // 8,
        "max_model_len": (2 + 256 // 8) * 8,

        # Skip cuda graph recording for fast test.
        "enforce_eager": True,

        # Required for spec decode.
        "use_v2_block_manager": True
    }])
@pytest.mark.parametrize("per_test_common_llm_kwargs", [
    {
        "model": "JackFram/llama-160m",
    },
])
@pytest.mark.parametrize("baseline_llm_kwargs", [{}])
@pytest.mark.parametrize("test_llm_kwargs", [
    {
        "speculative_model": "[ngram]",
        "num_speculative_tokens": 5,
        "ngram_prompt_lookup_max": 3,
    },
])
@pytest.mark.parametrize(
    "output_len",
    [
        # Use small output len for fast test.
        256,
    ])
@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("seed", [1])
def test_spec_decode_e2e_greedy_correctness_with_preemption(
        baseline_llm_generator, test_llm_generator, batch_size: int,
        output_len: int):
    """Verify greedy equality, even when some sequences are preempted mid-
    generation.
    """
    run_greedy_equality_correctness_test(baseline_llm_generator,
                                         test_llm_generator,
                                         batch_size,
                                         max_output_len=output_len,
                                         force_output_len=True)


@pytest.mark.parametrize(
    "common_llm_kwargs",
    [{
        "model": "JackFram/llama-68m",

        # Skip cuda graph recording for fast test.
        "enforce_eager": True,

        # Required for spec decode.
        "use_v2_block_manager": True
    }])
@pytest.mark.parametrize("per_test_common_llm_kwargs", [{}])
@pytest.mark.parametrize("baseline_llm_kwargs", [{}])
@pytest.mark.parametrize(
    "test_llm_kwargs",
    [
        {
            "speculative_model": "[ngram]",
            "num_speculative_tokens": k,
            "ngram_prompt_lookup_max": 3,
        }
        # Try a range of common k, as well as large speculation.
        for k in [1, 2, 3, 4, 5, 6, 7, 8, 9, 63]
    ] + [
        {
            "speculative_model": "[ngram]",
            "num_speculative_tokens": k,
            "ngram_prompt_lookup_max": 1,
        }
        # Try a range of common k, as well as large speculation.
        for k in [1, 2, 3, 4, 5, 6, 7, 8, 9, 63]
    ])
@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize(
    "output_len",
    [
        # Use smaller output len for fast test.
        32,
    ])
@pytest.mark.parametrize("seed", [1])
def test_many_k(baseline_llm_generator, test_llm_generator, batch_size: int,
                output_len: int):
    """Verify that speculative decoding produces exact equality to without spec
    decode with many different values of k.
    """
    run_greedy_equality_correctness_test(baseline_llm_generator,
                                         test_llm_generator,
                                         batch_size,
                                         max_output_len=output_len,
                                         force_output_len=True)


def run_greedy_equality_correctness_test(baseline_llm_generator,
                                         test_llm_generator,
                                         batch_size,
                                         max_output_len,
                                         force_output_len: bool,
                                         print_tokens: bool = False):
    """Helper method that compares the outputs of both the baseline LLM and
    the test LLM. It asserts greedy equality, e.g. that the outputs are exactly
    the same when temperature is zero.
    """
    temperature = 0.0

    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
        "San Francisco is know for its",
        "Facebook was created in 2004 by",
        "Curious George is a",
        "Python 3.11 brings improvements to its",
    ]

    prompts = [prompt for prompt, _ in zip(cycle(prompts), range(batch_size))]

    # If the test requires that we generated max_output_len tokens, then set the
    # sampling params to ignore eos token.
    ignore_eos = force_output_len

    sampling_params = SamplingParams(
        max_tokens=max_output_len,
        ignore_eos=ignore_eos,
        temperature=temperature,
    )

    spec_batch_tokens, spec_batch_token_ids = get_output_from_llm_generator(
        test_llm_generator, prompts, sampling_params)

    (baseline_batch_tokens,
     baseline_batch_token_ids) = get_output_from_llm_generator(
         baseline_llm_generator, prompts, sampling_params)

    assert len(baseline_batch_token_ids) == len(prompts)
    assert len(spec_batch_token_ids) == len(prompts)

    for i, (baseline_token_ids, baseline_tokens, spec_token_ids,
            spec_tokens) in enumerate(
                zip(baseline_batch_token_ids, baseline_batch_tokens,
                    spec_batch_token_ids, spec_batch_tokens)):
        if print_tokens:
            print(f'{i=} {baseline_tokens=}')
            print(f'{i=}     {spec_tokens=}')
        print(f'{i=} {baseline_token_ids=}')
        print(f'{i=}     {spec_token_ids=}')
        assert baseline_token_ids == spec_token_ids
