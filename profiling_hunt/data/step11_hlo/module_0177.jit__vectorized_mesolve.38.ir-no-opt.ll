; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @loop_not_select_fusion(ptr noalias align 256 dereferenceable(8) %0, ptr noalias align 256 dereferenceable(8) %1, ptr noalias align 256 dereferenceable(1) %2, ptr noalias align 256 dereferenceable(4) %3) #0 {
  %5 = getelementptr inbounds [1 x double], ptr %0, i32 0, i32 0
  %6 = load double, ptr %5, align 8, !invariant.load !1
  %7 = getelementptr inbounds [1 x double], ptr %1, i32 0, i32 0
  %8 = load double, ptr %7, align 8, !invariant.load !1
  %9 = fcmp olt double %6, %8
  %10 = zext i1 %9 to i32
  %11 = icmp eq i32 %10, 0
  %12 = icmp eq i32 %10, 11
  %13 = or i1 %11, %12
  %14 = zext i1 %13 to i8
  %15 = icmp eq i8 %14, 0
  %16 = zext i1 %15 to i8
  %17 = getelementptr inbounds [1 x i8], ptr %2, i32 0, i32 0
  store i8 %16, ptr %17, align 1
  %18 = getelementptr inbounds [1 x i32], ptr %3, i32 0, i32 0
  store i32 %10, ptr %18, align 4
  ret void
}

attributes #0 = { "nvvm.reqntid"="1,1,1" }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{}
