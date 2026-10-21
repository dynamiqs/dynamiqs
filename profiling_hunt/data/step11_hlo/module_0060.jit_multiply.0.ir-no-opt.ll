; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @wrapped_multiply(ptr noalias align 16 dereferenceable(1008) %0, ptr noalias align 16 dereferenceable(1008) %1, ptr noalias align 256 dereferenceable(1008) %2) #0 {
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !1
  %5 = getelementptr inbounds [63 x { double, double }], ptr %0, i32 0, i32 %4
  %6 = load { double, double }, ptr %5, align 8, !invariant.load !2
  %7 = getelementptr inbounds [63 x { double, double }], ptr %1, i32 0, i32 %4
  %8 = load { double, double }, ptr %7, align 8, !invariant.load !2
  %9 = extractvalue { double, double } %6, 0
  %10 = extractvalue { double, double } %6, 1
  %11 = extractvalue { double, double } %8, 0
  %12 = extractvalue { double, double } %8, 1
  %13 = fmul double %9, %11
  %14 = fmul double %10, %12
  %15 = fmul double %10, %11
  %16 = fmul double %9, %12
  %17 = fsub double %13, %14
  %18 = fadd double %15, %16
  %19 = insertvalue { double, double } poison, double %17, 0
  %20 = insertvalue { double, double } %19, double %18, 1
  %21 = getelementptr inbounds [63 x { double, double }], ptr %2, i32 0, i32 %4
  store { double, double } %20, ptr %21, align 8
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

attributes #0 = { "nvvm.reqntid"="63,1,1" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 0, i32 63}
!2 = !{}
