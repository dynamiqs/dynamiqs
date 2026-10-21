; ModuleID = 'jit_squeeze_consts'
source_filename = "jit_squeeze_consts"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@buffer_for_constant_1_0 = local_unnamed_addr addrspace(1) constant [64 x i8] zeroinitializer, align 256

!llvm.module.flags = !{!0}

!0 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
