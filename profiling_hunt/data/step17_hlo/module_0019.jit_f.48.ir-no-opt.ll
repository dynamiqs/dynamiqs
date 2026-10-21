; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @loop_select_fusion_5(ptr noalias align 256 dereferenceable(8) %0, ptr noalias align 256 dereferenceable(4) %1) #0 {
  %3 = getelementptr inbounds [1 x double], ptr %0, i32 0, i32 0
  %4 = load double, ptr %3, align 8, !invariant.load !1
  %5 = fcmp olt double %4, 1.000000e-01
  %6 = zext i1 %5 to i32
  %7 = getelementptr inbounds [1 x i32], ptr %1, i32 0, i32 0
  store i32 %6, ptr %7, align 4
  ret void
}

attributes #0 = { "nvvm.reqntid"="1,1,1" }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{}
