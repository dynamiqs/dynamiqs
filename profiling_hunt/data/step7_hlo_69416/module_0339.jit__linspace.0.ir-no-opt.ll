; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @input_concatenate_fusion(ptr noalias align 16 dereferenceable(8) %0, ptr noalias align 16 dereferenceable(8) %1, ptr noalias align 256 dereferenceable(640) %2) #0 {
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !1
  %5 = sitofp i32 %4 to double
  %6 = fmul double %5, f0x3F89EC8E951033D9
  %7 = getelementptr inbounds [1 x double], ptr %1, i32 0, i32 0
  %8 = load double, ptr %7, align 8, !invariant.load !2
  %9 = fsub double 1.000000e+00, %6
  %10 = getelementptr inbounds [1 x double], ptr %0, i32 0, i32 0
  %11 = load double, ptr %10, align 8, !invariant.load !2
  %12 = fmul double %11, f0x3F89EC8E951033D9
  %13 = fmul double %8, %9
  %14 = fmul double %5, %12
  %15 = fadd double %13, %14
  %16 = getelementptr inbounds [80 x double], ptr %2, i32 0, i32 %4
  store double %15, ptr %16, align 8
  %17 = icmp eq i32 %4, 0
  br i1 %17, label %18, label %21

18:                                               ; preds = %3
  %19 = load double, ptr %10, align 8, !invariant.load !2
  %20 = getelementptr inbounds [80 x double], ptr %2, i32 0, i32 79
  store double %19, ptr %20, align 8
  br label %21

21:                                               ; preds = %18, %3
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

attributes #0 = { "nvvm.reqntid"="79,1,1" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 0, i32 79}
!2 = !{}
