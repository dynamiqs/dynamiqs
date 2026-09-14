; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @input_concatenate_fusion(ptr noalias align 16 dereferenceable(8) %0, ptr noalias align 16 dereferenceable(8) %1, ptr noalias align 256 dereferenceable(1280) %2) #0 {
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !1
  %5 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %6 = mul i32 %4, 128
  %7 = add i32 %6, %5
  %8 = icmp sle i32 %7, 158
  br i1 %8, label %9, label %22

9:                                                ; preds = %3
  %10 = sitofp i32 %7 to double
  %11 = fmul double %10, f0x3F79C2D14EE4A102
  %12 = getelementptr inbounds [1 x double], ptr %1, i32 0, i32 0
  %13 = load double, ptr %12, align 8, !invariant.load !3
  %14 = fsub double 1.000000e+00, %11
  %15 = getelementptr inbounds [1 x double], ptr %0, i32 0, i32 0
  %16 = load double, ptr %15, align 8, !invariant.load !3
  %17 = fmul double %16, f0x3F79C2D14EE4A102
  %18 = fmul double %13, %14
  %19 = fmul double %10, %17
  %20 = fadd double %18, %19
  %21 = getelementptr inbounds [160 x double], ptr %2, i32 0, i32 %7
  store double %20, ptr %21, align 8
  br label %22

22:                                               ; preds = %9, %3
  %23 = icmp eq i32 %7, 0
  br i1 %23, label %24, label %28

24:                                               ; preds = %22
  %25 = getelementptr inbounds [1 x double], ptr %0, i32 0, i32 0
  %26 = load double, ptr %25, align 8, !invariant.load !3
  %27 = getelementptr inbounds [160 x double], ptr %2, i32 0, i32 159
  store double %26, ptr %27, align 8
  br label %28

28:                                               ; preds = %24, %22
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
!1 = !{i32 0, i32 2}
!2 = !{i32 0, i32 128}
!3 = !{}
