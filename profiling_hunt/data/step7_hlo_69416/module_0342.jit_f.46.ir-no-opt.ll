; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @loop_not_select_fusion(ptr noalias align 256 dereferenceable(8) %0, ptr noalias align 256 dereferenceable(1) %1, ptr noalias align 256 dereferenceable(4) %2) #0 {
  %4 = getelementptr inbounds [1 x double], ptr %0, i32 0, i32 0
  %5 = load double, ptr %4, align 8, !invariant.load !1
  %6 = fcmp olt double %5, 1.000000e-01
  %7 = zext i1 %6 to i32
  %8 = icmp eq i32 %7, 0
  %9 = icmp eq i32 %7, 11
  %10 = or i1 %8, %9
  %11 = zext i1 %10 to i8
  %12 = icmp eq i8 %11, 0
  %13 = zext i1 %12 to i8
  %14 = getelementptr inbounds [1 x i8], ptr %1, i32 0, i32 0
  store i8 %13, ptr %14, align 1
  %15 = getelementptr inbounds [1 x i32], ptr %2, i32 0, i32 0
  store i32 %7, ptr %15, align 4
  ret void
}

attributes #0 = { "nvvm.reqntid"="1,1,1" }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{}
