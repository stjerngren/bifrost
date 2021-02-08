
#
#    with autotvm.apply_history_best(log_file):
#        
#
#        # Generate the data to suse with both llvm and llvm stonne
#        data = np.random.uniform(-1, 1, size=data_shape).astype("float32")
#
#        target = "llvm -libs=stonne"
#        lib = relay.build_module.build(net, target, params=params)
#
#        ctx = tvm.context(target, 0)
#        module = runtime.GraphModule(lib["default"](ctx))
#        module.set_input("data", data)
#        module.run()
#        out_shape = (batch_size, out_channels, 10, 10)
#        out = module.get_output(0, tvm.nd.empty(out_shape))
#        out_stonne = out.asnumpy()
#        print(out_stonne)