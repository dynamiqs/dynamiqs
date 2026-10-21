; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @loop_multiply_fusion_2(ptr noalias align 256 dereferenceable(8) %0, ptr noalias align 256 dereferenceable(8) %1, ptr noalias align 256 dereferenceable(8) %2, ptr noalias align 256 dereferenceable(8) %3) #0 {
  %5 = getelementptr inbounds [1 x double], ptr %1, i32 0, i32 0
  %6 = load double, ptr %5, align 8, !invariant.load !1
  %7 = getelementptr inbounds [1 x double], ptr %0, i32 0, i32 0
  %8 = load double, ptr %7, align 8, !invariant.load !1
  %9 = fneg double %8
  %10 = getelementptr inbounds [1 x i64], ptr %2, i32 0, i32 0
  %11 = load i64, ptr %10, align 4, !invariant.load !1
  %12 = icmp eq i64 %11, 1
  %13 = fneg double %6
  %14 = select i1 %12, double %6, double %9
  %15 = select i1 %12, double %8, double %13
  %16 = sitofp i64 %11 to double
  %17 = fsub double %14, %15
  %18 = fmul double %16, %17
  %19 = getelementptr inbounds [1 x double], ptr %3, i32 0, i32 0
  store double %18, ptr %19, align 8
  ret void
}

attributes #0 = { "nvvm.reqntid"="1,1,1" }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{}
