; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @loop_pad_fusion(ptr noalias align 256 dereferenceable(65536) %0, ptr noalias align 256 dereferenceable(8) %1, ptr noalias align 256 dereferenceable(458752) %2) #0 {
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !1
  %5 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %6 = udiv i32 %4, 32
  %7 = icmp eq i32 %6, 0
  br i1 %7, label %8, label %26

8:                                                ; preds = %3
  %9 = getelementptr inbounds [1 x double], ptr %1, i32 0, i32 0
  %10 = load double, ptr %9, align 8, !invariant.load !3
  %11 = urem i32 %4, 32
  %12 = mul i32 %11, 128
  %13 = add i32 %12, %5
  %14 = getelementptr inbounds [4096 x { double, double }], ptr %0, i32 0, i32 %13
  %15 = load { double, double }, ptr %14, align 8, !invariant.load !3
  %16 = extractvalue { double, double } %15, 0
  %17 = extractvalue { double, double } %15, 1
  %18 = fmul double %10, %16
  %19 = fmul double %17, 0.000000e+00
  %20 = fmul double %16, 0.000000e+00
  %21 = fmul double %10, %17
  %22 = fsub double %18, %19
  %23 = fadd double %20, %21
  %24 = insertvalue { double, double } poison, double %22, 0
  %25 = insertvalue { double, double } %24, double %23, 1
  br label %27

26:                                               ; preds = %3
  br label %27

27:                                               ; preds = %8, %26
  %28 = phi { double, double } [ zeroinitializer, %26 ], [ %25, %8 ]
  br label %29

29:                                               ; preds = %27
  %30 = mul i32 %4, 128
  %31 = add i32 %30, %5
  %32 = getelementptr inbounds [28672 x { double, double }], ptr %2, i32 0, i32 %31
  store { double, double } %28, ptr %32, align 8
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
!1 = !{i32 0, i32 224}
!2 = !{i32 0, i32 128}
!3 = !{}
