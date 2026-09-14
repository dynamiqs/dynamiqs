; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @loop_select_fusion_2(ptr noalias align 256 dereferenceable(1) %0, ptr noalias align 256 dereferenceable(8) %1) #0 {
  %3 = getelementptr inbounds [1 x i8], ptr %0, i32 0, i32 0
  %4 = load i8, ptr %3, align 1, !invariant.load !1
  %5 = trunc i8 %4 to i1
  %6 = xor i1 %5, true
  %7 = zext i1 %6 to i64
  %8 = getelementptr inbounds [1 x i64], ptr %1, i32 0, i32 0
  store i64 %7, ptr %8, align 4
  ret void
}

attributes #0 = { "nvvm.reqntid"="1,1,1" }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{}
