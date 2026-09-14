; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @loop_select_fusion(ptr noalias nofree readonly align 256 captures(none) dereferenceable(65536) %0, ptr noalias nofree readonly align 256 captures(none) dereferenceable(1) %1, ptr noalias nofree writeonly align 256 captures(none) dereferenceable(65536) %2) local_unnamed_addr #0 {
  %4 = addrspacecast ptr %1 to ptr addrspace(1)
  %5 = addrspacecast ptr %0 to ptr addrspace(1)
  %6 = addrspacecast ptr %2 to ptr addrspace(1)
  %7 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !2
  %8 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !3
  %9 = load i8, ptr addrspace(1) %4, align 256, !invariant.load !4
  %10 = shl nuw nsw i32 %7, 7
  %11 = or disjoint i32 %10, %8
  %12 = zext nneg i32 %11 to i64
  %13 = getelementptr inbounds [16 x i8], ptr addrspace(1) %5, i64 %12
  %14 = load <2 x double>, ptr addrspace(1) %13, align 16, !invariant.load !4
  %.unpack5 = extractelement <2 x double> %14, i32 0
  %.unpack26 = extractelement <2 x double> %14, i32 1
  %15 = trunc i8 %9 to i1
  %16 = getelementptr inbounds [16 x i8], ptr addrspace(1) %6, i64 %12
  %.elt = select i1 %15, double 0.000000e+00, double %.unpack5
  %.elt4 = select i1 %15, double 0.000000e+00, double %.unpack26
  %17 = insertelement <2 x double> poison, double %.elt, i32 0
  %18 = insertelement <2 x double> %17, double %.elt4, i32 1
  store <2 x double> %18, ptr addrspace(1) %16, align 16
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) "nvvm.reqntid"="128,1,1" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!2 = !{i32 0, i32 32}
!3 = !{i32 0, i32 128}
!4 = !{}
