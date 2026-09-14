; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @wrapped_dynamic_slice(ptr noalias align 16 dereferenceable(2048) %0, ptr noalias align 16 dereferenceable(8) %1, ptr noalias align 16 dereferenceable(8) %2, ptr noalias align 256 dereferenceable(32) %3) #0 {
  %5 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !1
  %6 = sext i32 %5 to i64
  %7 = getelementptr inbounds [1 x i64], ptr %1, i32 0, i32 0
  %8 = load i64, ptr %7, align 4, !invariant.load !2
  %9 = call i64 @llvm.smin.i64(i64 %8, i64 1)
  %10 = call i64 @llvm.smax.i64(i64 %9, i64 0)
  %11 = getelementptr inbounds [1 x i64], ptr %2, i32 0, i32 0
  %12 = load i64, ptr %11, align 4, !invariant.load !2
  %13 = call i64 @llvm.smin.i64(i64 %12, i64 62)
  %14 = call i64 @llvm.smax.i64(i64 %13, i64 0)
  %15 = add i64 %6, %14
  %16 = mul i64 %10, 64
  %17 = add i64 %16, %15
  %18 = getelementptr inbounds [128 x { double, double }], ptr %0, i32 0, i64 %17
  %19 = load { double, double }, ptr %18, align 8, !invariant.load !2
  %20 = getelementptr inbounds [2 x { double, double }], ptr %3, i32 0, i64 %6
  store { double, double } %19, ptr %20, align 8
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.smin.i64(i64, i64) #2

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.smax.i64(i64, i64) #2

attributes #0 = { "nvvm.reqntid"="2,1,1" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 0, i32 2}
!2 = !{}
