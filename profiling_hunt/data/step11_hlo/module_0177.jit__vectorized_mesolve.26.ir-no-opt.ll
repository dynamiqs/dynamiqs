; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @loop_compare_fusion(ptr noalias align 256 dereferenceable(8) %0, ptr noalias align 256 dereferenceable(1) %1) #0 {
  %3 = getelementptr inbounds [1 x i64], ptr %0, i32 0, i32 0
  %4 = load i64, ptr %3, align 4, !invariant.load !1
  %5 = srem i64 %4, 20
  %6 = icmp slt i64 %5, 0
  %7 = icmp ne i64 %5, 0
  %8 = and i1 %6, %7
  %9 = add i64 %5, 20
  %10 = select i1 %8, i64 %9, i64 %5
  %11 = icmp eq i64 %10, 0
  %12 = zext i1 %11 to i8
  %13 = getelementptr inbounds [1 x i8], ptr %1, i32 0, i32 0
  store i8 %12, ptr %13, align 1
  ret void
}

attributes #0 = { "nvvm.reqntid"="1,1,1" }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{}
