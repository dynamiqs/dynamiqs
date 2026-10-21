; ModuleID = 'jit_f_consts'
source_filename = "jit_f_consts"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@buffer_for_constant_266_0 = local_unnamed_addr addrspace(1) constant [64 x i8] c"{\14\AEG\E1z\84?\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00", align 256
@buffer_for_constant_260_0 = local_unnamed_addr addrspace(1) constant [64 x i8] c"\9A\99\99\99\99\99\B9?\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00", align 256
@buffer_for_constant_256_0 = local_unnamed_addr addrspace(1) constant [64 x i8] zeroinitializer, align 256
@buffer_for_constant_4_0 = local_unnamed_addr addrspace(1) constant [64 x i8] c"\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00", align 256
@buffer_for_constant_3_0 = local_unnamed_addr addrspace(1) constant [64 x i8] zeroinitializer, align 256
@buffer_for_constant_164_0 = local_unnamed_addr addrspace(1) constant [64 x i8] c"\A0\86\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00", align 256
@buffer_for_constant_2828_0 = local_unnamed_addr addrspace(1) constant [64 x i8] zeroinitializer, align 256
@buffer_for_constant_2833_0 = local_unnamed_addr addrspace(1) global [112 x i8] zeroinitializer, align 256
@buffer_for_constant_3047_0 = local_unnamed_addr addrspace(1) global [64 x i8] zeroinitializer, align 256
@buffer_for_constant_3044_0 = local_unnamed_addr addrspace(1) global [65536 x i8] zeroinitializer, align 256
@buffer_for_constant_3041_0 = local_unnamed_addr addrspace(1) global [784 x i8] zeroinitializer, align 256
@buffer_for_constant_2682__sunk2_7_0 = local_unnamed_addr addrspace(1) global [320 x i8] zeroinitializer, align 256
@buffer_for_constant_2682__sunk2_8_0 = local_unnamed_addr addrspace(1) global [320 x i8] zeroinitializer, align 256
@buffer_for_constant_3012_0 = local_unnamed_addr addrspace(1) global [64 x i8] zeroinitializer, align 256

!llvm.module.flags = !{!0}

!0 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
