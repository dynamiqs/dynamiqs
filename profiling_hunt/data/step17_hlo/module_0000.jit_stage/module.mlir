#loc1 = loc("args[0]")
module @jit_stage attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<i64> loc("args[0]")) -> (tensor<i64> {jax.result_info = "result"}) {
    return %arg0 : tensor<i64> loc(#loc4)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("/srv/data/users/nlepage/new_dynamiqs/dynamiqs/profiling_hunt/scripts/step17_dump_bare.py":24:22 to :39)
#loc3 = loc("<module>"(#loc2))
#loc4 = loc("jit(stage)"(#loc3))
