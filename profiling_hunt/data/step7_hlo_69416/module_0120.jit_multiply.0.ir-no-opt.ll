; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @loop_multiply_fusion(ptr noalias align 16 dereferenceable(1024) %0, ptr noalias align 16 dereferenceable(8) %1, ptr noalias align 256 dereferenceable(1024) %2) #0 {
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !1
  %5 = getelementptr inbounds [1 x double], ptr %1, i32 0, i32 0
  %6 = load double, ptr %5, align 8, !invariant.load !2
  %7 = getelementptr inbounds [64 x { double, double }], ptr %0, i32 0, i32 %4
  %8 = load { double, double }, ptr %7, align 8, !invariant.load !2
  %9 = extractvalue { double, double } %8, 0
  %10 = extractvalue { double, double } %8, 1
  %11 = fmul double %6, %9
  %12 = fmul double %10, 0.000000e+00
  %13 = fmul double %9, 0.000000e+00
  %14 = fmul double %6, %10
  %15 = fsub double %11, %12
  %16 = fadd double %13, %14
  %17 = insertvalue { double, double } poison, double %15, 0
  %18 = insertvalue { double, double } %17, double %16, 1
  %19 = getelementptr inbounds [64 x { double, double }], ptr %2, i32 0, i32 %4
  store { double, double } %18, ptr %19, align 8
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

attributes #0 = { "nvvm.reqntid"="64,1,1" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 0, i32 64}
!2 = !{}
