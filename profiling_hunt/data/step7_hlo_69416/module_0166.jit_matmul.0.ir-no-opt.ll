; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @loop_multiply_fusion(ptr noalias align 16 dereferenceable(1024) %0, ptr noalias align 16 dereferenceable(1024) %1, ptr noalias align 256 dereferenceable(65536) %2) #0 {
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !1
  %5 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %6 = mul i32 %4, 2
  %7 = udiv i32 %5, 64
  %8 = add i32 %6, %7
  %9 = urem i32 %5, 64
  %10 = getelementptr inbounds [64 x { double, double }], ptr %0, i32 0, i32 %8
  %11 = load { double, double }, ptr %10, align 8, !invariant.load !3
  %12 = getelementptr inbounds [64 x { double, double }], ptr %1, i32 0, i32 %9
  %13 = load { double, double }, ptr %12, align 8, !invariant.load !3
  %14 = extractvalue { double, double } %11, 0
  %15 = extractvalue { double, double } %11, 1
  %16 = extractvalue { double, double } %13, 0
  %17 = extractvalue { double, double } %13, 1
  %18 = fmul double %14, %16
  %19 = fmul double %15, %17
  %20 = fmul double %15, %16
  %21 = fmul double %14, %17
  %22 = fsub double %18, %19
  %23 = fadd double %20, %21
  %24 = insertvalue { double, double } poison, double %22, 0
  %25 = insertvalue { double, double } %24, double %23, 1
  %26 = mul i32 %4, 128
  %27 = add i32 %26, %5
  %28 = getelementptr inbounds [4096 x { double, double }], ptr %2, i32 0, i32 %27
  store { double, double } %25, ptr %28, align 8
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

attributes #0 = { "nvvm.reqntid"="128,1,1" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 0, i32 32}
!2 = !{i32 0, i32 128}
!3 = !{}
