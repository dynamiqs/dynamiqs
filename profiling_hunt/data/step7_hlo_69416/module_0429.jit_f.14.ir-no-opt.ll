; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @wrapped_subtract(ptr noalias align 256 dereferenceable(8) %0, ptr noalias align 256 dereferenceable(8) %1, ptr noalias align 256 dereferenceable(8) %2) #0 {
  %4 = getelementptr inbounds [1 x double], ptr %0, i32 0, i32 0
  %5 = load double, ptr %4, align 8, !invariant.load !1
  %6 = getelementptr inbounds [1 x double], ptr %1, i32 0, i32 0
  %7 = load double, ptr %6, align 8, !invariant.load !1
  %8 = fsub double %5, %7
  %9 = getelementptr inbounds [1 x double], ptr %2, i32 0, i32 0
  store double %8, ptr %9, align 8
  ret void
}

attributes #0 = { "nvvm.reqntid"="1,1,1" }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{}
