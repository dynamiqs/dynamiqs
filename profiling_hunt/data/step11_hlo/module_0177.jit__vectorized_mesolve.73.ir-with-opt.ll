; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @loop_select_fusion_1(ptr noalias nofree readonly align 256 captures(none) dereferenceable(65536) %0, ptr noalias nofree align 256 captures(none) dereferenceable(65536) %1, ptr noalias nofree readonly align 256 captures(none) dereferenceable(1) %2, ptr noalias nofree readnone align 256 captures(none) dereferenceable(65536) %3) local_unnamed_addr #0 {
  %5 = addrspacecast ptr %2 to ptr addrspace(1)
  %6 = addrspacecast ptr %0 to ptr addrspace(1)
  %7 = addrspacecast ptr %1 to ptr addrspace(1)
  %8 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !2
  %9 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !3
  %10 = load i8, ptr addrspace(1) %5, align 256, !invariant.load !4
  %11 = shl nuw nsw i32 %8, 7
  %12 = or disjoint i32 %11, %9
  %13 = zext nneg i32 %12 to i64
  %14 = getelementptr inbounds [16 x i8], ptr addrspace(1) %6, i64 %13
  %15 = load <2 x double>, ptr addrspace(1) %14, align 16, !invariant.load !4
  %.unpack8 = extractelement <2 x double> %15, i32 0
  %.unpack29 = extractelement <2 x double> %15, i32 1
  %16 = getelementptr inbounds [16 x i8], ptr addrspace(1) %7, i64 %13
  %17 = load <2 x double>, ptr addrspace(1) %16, align 16
  %.unpack310 = extractelement <2 x double> %17, i32 0
  %.unpack511 = extractelement <2 x double> %17, i32 1
  %18 = trunc i8 %10 to i1
  %.elt = select i1 %18, double %.unpack8, double %.unpack310
  %.elt7 = select i1 %18, double %.unpack29, double %.unpack511
  %19 = insertelement <2 x double> poison, double %.elt, i32 0
  %20 = insertelement <2 x double> %19, double %.elt7, i32 1
  store <2 x double> %20, ptr addrspace(1) %16, align 16
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
