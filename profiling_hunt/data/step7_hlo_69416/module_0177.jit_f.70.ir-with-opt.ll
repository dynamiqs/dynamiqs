; ModuleID = 'jit_f_consts'
source_filename = "jit_f_consts"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@buffer_for_constant_202_0 = local_unnamed_addr addrspace(1) constant [64 x i8] zeroinitializer, align 256
@buffer_for_constant_24_0 = local_unnamed_addr addrspace(1) constant [64 x i8] zeroinitializer, align 256
@buffer_for_constant_1823_0 = local_unnamed_addr addrspace(1) constant [64 x i8] zeroinitializer, align 256
@buffer_for_constant_1828_0 = local_unnamed_addr addrspace(1) constant [64 x i8] c"{\14\AEG\E1z\84?\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00", align 256
@buffer_for_constant_242_0 = local_unnamed_addr addrspace(1) constant [64 x i8] c"\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00", align 256
@buffer_for_constant_4993_0 = local_unnamed_addr addrspace(1) constant [64 x i8] zeroinitializer, align 256
@buffer_for_constant_251__sunk2_5_0 = local_unnamed_addr addrspace(1) global [784 x i8] zeroinitializer, align 256
@buffer_for_constant_5260_0 = local_unnamed_addr addrspace(1) global [992 x i8] zeroinitializer, align 256
@buffer_for_constant_5261_0 = local_unnamed_addr addrspace(1) global [1008 x i8] zeroinitializer, align 256
@buffer_for_constant_5262_0 = local_unnamed_addr addrspace(1) global [1008 x i8] zeroinitializer, align 256
@buffer_for_constant_5265_0 = local_unnamed_addr addrspace(1) global [512 x i8] zeroinitializer, align 256
@buffer_for_constant_5266_0 = local_unnamed_addr addrspace(1) global [992 x i8] zeroinitializer, align 256
@buffer_for_constant_5269_0 = local_unnamed_addr addrspace(1) global [1536 x i8] zeroinitializer, align 256
@buffer_for_constant_5256_0 = local_unnamed_addr addrspace(1) global [64 x i8] zeroinitializer, align 256
@buffer_for_constant_253__sunk_7_0 = local_unnamed_addr addrspace(1) global [112 x i8] zeroinitializer, align 256
@buffer_for_constant_1815__sunk2_7_0 = local_unnamed_addr addrspace(1) global [80 x i8] zeroinitializer, align 256
@buffer_for_constant_1815__sunk2_8_0 = local_unnamed_addr addrspace(1) global [80 x i8] zeroinitializer, align 256
@buffer_for_constant_5209_0 = local_unnamed_addr addrspace(1) global [64 x i8] zeroinitializer, align 256
@buffer_for_constant_1815_0 = local_unnamed_addr addrspace(1) global [80 x i8] zeroinitializer, align 256
@buffer_for_constant_234_0 = local_unnamed_addr addrspace(1) constant [64 x i8] c"\8D\ED\B5\A0\F7\C6\B0>\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00", align 256
@buffer_for_constant_233_0 = local_unnamed_addr addrspace(1) constant [64 x i8] c"\CD\CC\CC\CC\CC\CC\EC?\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00", align 256
@buffer_for_constant_232_0 = local_unnamed_addr addrspace(1) constant [64 x i8] c"\9A\99\99\99\99\99\C9?\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00", align 256
@buffer_for_constant_231_0 = local_unnamed_addr addrspace(1) constant [64 x i8] c"\00\00\00\00\00\00\14@\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00", align 256
@buffer_for_constant_262_0 = local_unnamed_addr addrspace(1) constant [64 x i8] c"\10'\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00", align 256
@buffer_for_constant_263_0 = local_unnamed_addr addrspace(1) constant [64 x i8] zeroinitializer, align 256

!llvm.module.flags = !{!0}

!0 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
