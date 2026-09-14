; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @loop_and_fusion_1(ptr noalias align 256 dereferenceable(1) %0, ptr noalias align 256 dereferenceable(8) %1, ptr noalias align 256 dereferenceable(160) %2, ptr noalias align 256 dereferenceable(8) %3, ptr noalias align 256 dereferenceable(8) %4, ptr noalias align 256 dereferenceable(1) %5) #0 {
  %7 = getelementptr inbounds [1 x i64], ptr %4, i32 0, i32 0
  %8 = load i64, ptr %7, align 4, !invariant.load !1
  %9 = getelementptr inbounds [1 x i64], ptr %3, i32 0, i32 0
  %10 = load i64, ptr %9, align 4, !invariant.load !1
  %11 = add i64 %8, %10
  %12 = icmp slt i64 %11, 0
  %13 = add i64 %11, 20
  %14 = select i1 %12, i64 %13, i64 %11
  %15 = call i64 @llvm.smin.i64(i64 %14, i64 19)
  %16 = call i64 @llvm.smax.i64(i64 %15, i64 0)
  %17 = getelementptr inbounds [20 x double], ptr %2, i32 0, i64 %16
  %18 = load double, ptr %17, align 8, !invariant.load !1
  %19 = getelementptr inbounds [1 x double], ptr %1, i32 0, i32 0
  %20 = load double, ptr %19, align 8, !invariant.load !1
  %21 = getelementptr inbounds [1 x i8], ptr %0, i32 0, i32 0
  %22 = load i8, ptr %21, align 1, !invariant.load !1
  %23 = fcmp ole double %18, %20
  %24 = zext i1 %23 to i8
  %25 = and i8 %22, %24
  %26 = icmp slt i64 %11, 20
  %27 = zext i1 %26 to i8
  %28 = and i8 %25, %27
  %29 = icmp slt i64 %10, 20
  %30 = zext i1 %29 to i8
  %31 = and i8 %28, %30
  %32 = getelementptr inbounds [1 x i8], ptr %5, i32 0, i32 0
  store i8 %31, ptr %32, align 1
  ret void
}

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.smin.i64(i64, i64) #1

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.smax.i64(i64, i64) #1

attributes #0 = { "nvvm.reqntid"="1,1,1" }
attributes #1 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{}
