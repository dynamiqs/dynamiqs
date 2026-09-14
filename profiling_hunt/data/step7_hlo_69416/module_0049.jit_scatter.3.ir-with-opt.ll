; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @loop_dynamic_slice_fusion(ptr noalias nofree readonly align 256 captures(none) dereferenceable(1024) %0, ptr noalias nofree readonly align 16 captures(none) dereferenceable(8) %1, ptr noalias nofree writeonly align 256 captures(none) dereferenceable(1008) %2) local_unnamed_addr #0 {
  %4 = addrspacecast ptr %1 to ptr addrspace(1)
  %5 = addrspacecast ptr %0 to ptr addrspace(1)
  %6 = addrspacecast ptr %2 to ptr addrspace(1)
  %7 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %8 = getelementptr inbounds i8, ptr addrspace(1) %4, i64 4
  %9 = load i32, ptr addrspace(1) %8, align 4, !invariant.load !3
  %10 = icmp sgt i32 %9, 0
  %11 = zext i1 %10 to i32
  %12 = add nuw nsw i32 %7, %11
  %13 = zext nneg i32 %12 to i64
  %14 = getelementptr inbounds [16 x i8], ptr addrspace(1) %5, i64 %13
  %15 = load <2 x double>, ptr addrspace(1) %14, align 16, !invariant.load !3
  %.unpack5 = extractelement <2 x double> %15, i32 0
  %.unpack26 = extractelement <2 x double> %15, i32 1
  %16 = zext nneg i32 %7 to i64
  %17 = getelementptr inbounds [16 x i8], ptr addrspace(1) %6, i64 %16
  %18 = insertelement <2 x double> poison, double %.unpack5, i32 0
  %19 = insertelement <2 x double> %18, double %.unpack26, i32 1
  store <2 x double> %19, ptr addrspace(1) %17, align 16
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) "nvvm.reqntid"="63,1,1" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!2 = !{i32 0, i32 63}
!3 = !{}
