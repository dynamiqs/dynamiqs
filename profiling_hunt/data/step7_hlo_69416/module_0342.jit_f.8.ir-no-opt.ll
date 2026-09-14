; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @loop_broadcast_fusion_1(ptr noalias align 256 dereferenceable(5242880) %0) #0 {
  %2 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !1
  %3 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %4 = mul i32 %3, 4
  %5 = mul i32 %2, 512
  %6 = add i32 %4, %5
  %7 = getelementptr inbounds [327680 x { double, double }], ptr %0, i32 0, i32 %6
  store { double, double } { double +inf, double 0.000000e+00 }, ptr %7, align 8
  %8 = add i32 %6, 1
  %9 = getelementptr inbounds [327680 x { double, double }], ptr %0, i32 0, i32 %8
  store { double, double } { double +inf, double 0.000000e+00 }, ptr %9, align 8
  %10 = add i32 %6, 2
  %11 = getelementptr inbounds [327680 x { double, double }], ptr %0, i32 0, i32 %10
  store { double, double } { double +inf, double 0.000000e+00 }, ptr %11, align 8
  %12 = add i32 %6, 3
  %13 = getelementptr inbounds [327680 x { double, double }], ptr %0, i32 0, i32 %12
  store { double, double } { double +inf, double 0.000000e+00 }, ptr %13, align 8
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
!1 = !{i32 0, i32 640}
!2 = !{i32 0, i32 128}
