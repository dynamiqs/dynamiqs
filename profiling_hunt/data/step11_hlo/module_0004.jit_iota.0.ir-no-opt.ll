; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @wrapped_iota(ptr noalias align 256 dereferenceable(1008) %0) #0 {
  %2 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !1
  %3 = sitofp i32 %2 to double
  %4 = insertvalue { double, double } poison, double %3, 0
  %5 = insertvalue { double, double } %4, double 0.000000e+00, 1
  %6 = getelementptr inbounds [63 x { double, double }], ptr %0, i32 0, i32 %2
  store { double, double } %5, ptr %6, align 8
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

attributes #0 = { "nvvm.reqntid"="63,1,1" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 0, i32 63}
