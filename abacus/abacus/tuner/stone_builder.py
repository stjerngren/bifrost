import shutil
import time 
import os

from tvm.autotvm.measure.measure import MeasureResult, MeasureErrorNo, Builder
from tvm.autotvm.measure.measure_methods import BuildResult, _build_func_common
from tvm.autotvm.measure.local_executor import LocalExecutor
from tvm.autotvm.task.space import InstantiationError
from tvm.contrib import tar
import tempfile


class StonneLocalBuilder(Builder):
    """Run compilation on local machine

    Parameters
    ----------
    timeout: float
        The timeout of a compilation
    n_parallel: int
        The number of tasks run in parallel. "None" will use all cpu cores
    build_func: callable or str
        If is 'default', use default build function
        If is 'ndk', use function for android ndk
        If is callable, use it as custom build function, expect lib_format field.
    """

    def __init__(self, timeout=10, n_parallel=None, build_func="default"):
        super(StonneLocalBuilder, self).__init__(timeout, n_parallel)

        if isinstance(build_func, str):
            build_func = tar.tar

        self.build_func = _StonneWrappedBuildFunc(build_func)
        self.executor = LocalExecutor(timeout=timeout)
        self.tmp_dir = tempfile.mkdtemp()

    def build(self, measure_inputs):
        results = []

        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        self.tmp_dir = tempfile.mkdtemp()

        for i in range(0, len(measure_inputs), self.n_parallel):
            futures = []
            for inp in measure_inputs[i : i + self.n_parallel]:
                ret = self.executor.submit(self.build_func, inp, self.tmp_dir, **self.build_kwargs)
                futures.append(ret)

            for future in futures:
                res = future.get()

                if isinstance(res, Exception):
                    # timeout or fleet error, return MeasureResult directly
                    results.append(
                        MeasureResult(
                            (res,), MeasureErrorNo.BUILD_TIMEOUT, self.timeout, time.time()
                        )
                    )
                elif res.error is not None:
                    # instantiation error
                    if isinstance(res.error, InstantiationError):
                        results.append(
                            MeasureResult(
                                (res.error,),
                                MeasureErrorNo.INSTANTIATION_ERROR,
                                res.time_cost,
                                time.time(),
                            )
                        )
                    else:
                        if "InstantiationError" in str(res.error):
                            msg = str(res.error)
                            try:
                                msg = msg.split("\n")[-2].split(": ")[1]
                            except Exception:  # pylint: disable=broad-except
                                pass
                            results.append(
                                MeasureResult(
                                    (InstantiationError(msg),),
                                    MeasureErrorNo.INSTANTIATION_ERROR,
                                    res.time_cost,
                                    time.time(),
                                )
                            )
                        else:  # tvm error
                            results.append(
                                MeasureResult(
                                    (res.error,),
                                    MeasureErrorNo.COMPILE_HOST,
                                    res.time_cost,
                                    time.time(),
                                )
                            )
                else:
                    # return BuildResult
                    results.append(res)

        return results


class _StonneWrappedBuildFunc:
    """
    Wrap build_func to a function that can be used in measure.

    Note: this is a class instead of a closure so that it can be pickled when
    using multiprocessing.

    Parameters
    ----------
    build_func : The compilation function
        We expect fcompile to contain an attr "output_format"

    Returns
    -------
    wrapped_build_func : callable
        The wrapped build function
    """

    def __init__(self, build_func):
        if not hasattr(build_func, "output_format"):
            raise AttributeError("Expect build_func to have the attribute output_format.")
        self.build_func = build_func

    def __call__(self, measure_input, tmp_dir, **kwargs):
        """
        Wrapped build func.

        Parameters
        ----------
        measure_input: MeasureInput
            The input of measurement

        tmp_dir: str
            The path of temporary directory to export generated library
        """
        tic = time.time()
        try:
            # Find library based on relative paths
            # TODO: Modify this so that several fucntions can be uploaded
            dirname = os.path.dirname(__file__)
            filename = os.path.join(dirname, "../../lib/conv_forward.so")

            from random import getrandbits

            filename = os.path.join(
                tmp_dir, "tmp_func_%0x.%s" % (getrandbits(64), self.build_func.output_format)
            )

            func, arg_info = _build_func_common(measure_input, **kwargs)
            func.export_library(filename, self.build_func)
        except Exception as e:  # pylint: disable=broad-except
            return BuildResult(None, None, e, time.time() - tic)
        return BuildResult(filename, arg_info, None, time.time() - tic)



