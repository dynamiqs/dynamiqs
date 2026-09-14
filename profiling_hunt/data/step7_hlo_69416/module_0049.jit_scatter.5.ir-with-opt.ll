; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @loop_dynamic_update_slice_fusion(ptr noalias nofree writeonly align 256 captures(none) dereferenceable(1024) %0, ptr noalias nofree readonly align 256 captures(none) dereferenceable(1008) %1, ptr noalias nofree readonly align 16 captures(none) dereferenceable(1008) %2, ptr noalias nofree readonly align 256 captures(none) dereferenceable(1) %3, ptr noalias nofree readonly align 16 captures(none) dereferenceable(8) %4, ptr noalias nofree readnone align 256 captures(none) dereferenceable(1024) %5) local_unnamed_addr #0 {
  %7 = addrspacecast ptr %4 to ptr addrspace(1)
  %8 = addrspacecast ptr %3 to ptr addrspace(1)
  %9 = addrspacecast ptr %2 to ptr addrspace(1)
  %10 = addrspacecast ptr %1 to ptr addrspace(1)
  %11 = addrspacecast ptr %0 to ptr addrspace(1)
  %12 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %13 = getelementptr inbounds i8, ptr addrspace(1) %7, i64 4
  %14 = load i32, ptr addrspace(1) %13, align 4, !invariant.load !3
  %15 = icmp sgt i32 %14, 0
  %16 = zext i1 %15 to i32
  %17 = add nuw nsw i32 %12, %16
  %18 = load i8, ptr addrspace(1) %8, align 256, !invariant.load !3
  %19 = zext nneg i32 %12 to i64
  %20 = getelementptr inbounds [16 x i8], ptr addrspace(1) %9, i64 %19
  %21 = load <2 x double>, ptr addrspace(1) %20, align 16, !invariant.load !3
  %.unpack8 = extractelement <2 x double> %21, i32 0
  %.unpack29 = extractelement <2 x double> %21, i32 1
  %22 = getelementptr inbounds [16 x i8], ptr addrspace(1) %10, i64 %19
  %23 = load <2 x double>, ptr addrspace(1) %22, align 16, !invariant.load !3
  %.unpack310 = extractelement <2 x double> %23, i32 0
  %.unpack511 = extractelement <2 x double> %23, i32 1
  %24 = trunc i8 %18 to i1
  %25 = zext nneg i32 %17 to i64
  %26 = getelementptr inbounds [16 x i8], ptr addrspace(1) %11, i64 %25
  %.elt = select i1 %24, double %.unpack8, double %.unpack310
  %.elt7 = select i1 %24, double %.unpack29, double %.unpack511
  %27 = insertelement <2 x double> poison, double %.elt, i32 0
  %28 = insertelement <2 x double> %27, double %.elt7, i32 1
  store <2 x double> %28, ptr addrspace(1) %26, align 16
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
