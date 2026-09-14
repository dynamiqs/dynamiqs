; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write)
define ptx_kernel void @loop_broadcast_fusion_1(ptr noalias nofree writeonly align 256 captures(none) dereferenceable(10485760) %0) local_unnamed_addr #0 {
  %2 = addrspacecast ptr %0 to ptr addrspace(1)
  %3 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !2
  %4 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !3
  %5 = shl nuw nsw i32 %4, 2
  %6 = shl nuw nsw i32 %3, 9
  %7 = or disjoint i32 %5, %6
  %8 = zext nneg i32 %7 to i64
  %9 = getelementptr inbounds [16 x i8], ptr addrspace(1) %2, i64 %8
  store <2 x double> <double +inf, double 0.000000e+00>, ptr addrspace(1) %9, align 64
  %10 = getelementptr inbounds i8, ptr addrspace(1) %9, i64 16
  store <2 x double> <double +inf, double 0.000000e+00>, ptr addrspace(1) %10, align 16
  %11 = getelementptr inbounds i8, ptr addrspace(1) %9, i64 32
  store <2 x double> <double +inf, double 0.000000e+00>, ptr addrspace(1) %11, align 32
  %12 = getelementptr inbounds i8, ptr addrspace(1) %9, i64 48
  store <2 x double> <double +inf, double 0.000000e+00>, ptr addrspace(1) %12, align 16
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) "nvvm.reqntid"="128,1,1" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!2 = !{i32 0, i32 1280}
!3 = !{i32 0, i32 128}
