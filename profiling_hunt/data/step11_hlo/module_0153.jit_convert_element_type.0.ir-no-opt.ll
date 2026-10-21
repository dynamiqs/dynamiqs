; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @wrapped_convert(ptr noalias align 16 dereferenceable(8) %0, ptr noalias align 256 dereferenceable(16) %1) #0 {
  %3 = getelementptr inbounds [1 x double], ptr %0, i32 0, i32 0
  %4 = load double, ptr %3, align 8, !invariant.load !1
  %5 = insertvalue { double, double } poison, double %4, 0
  %6 = insertvalue { double, double } %5, double 0.000000e+00, 1
  %7 = getelementptr inbounds [1 x { double, double }], ptr %1, i32 0, i32 0
  store { double, double } %6, ptr %7, align 8
  ret void
}

attributes #0 = { "nvvm.reqntid"="1,1,1" }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{}
