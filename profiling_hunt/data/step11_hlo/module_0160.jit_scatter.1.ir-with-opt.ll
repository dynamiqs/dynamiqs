; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @loop_dynamic_update_slice_fusion(ptr noalias nofree writeonly align 256 captures(none) dereferenceable(1024) %0, ptr noalias nofree readonly align 16 captures(none) dereferenceable(4) %1, ptr noalias nofree readonly align 256 captures(none) dereferenceable(16) %2, ptr noalias nofree readonly align 16 captures(none) dereferenceable(16) %3, ptr noalias nofree readonly align 256 captures(none) dereferenceable(1) %4, ptr noalias nofree readnone align 256 captures(none) dereferenceable(1024) %5) local_unnamed_addr #0 {
  %7 = addrspacecast ptr %1 to ptr addrspace(1)
  %8 = addrspacecast ptr %4 to ptr addrspace(1)
  %9 = addrspacecast ptr %3 to ptr addrspace(1)
  %10 = addrspacecast ptr %2 to ptr addrspace(1)
  %11 = addrspacecast ptr %0 to ptr addrspace(1)
  %12 = load i32, ptr addrspace(1) %7, align 16, !invariant.load !2
  %13 = tail call i32 @llvm.smax.i32(i32 %12, i32 0)
  %14 = tail call i32 @llvm.umin.i32(i32 %13, i32 63)
  %15 = load i8, ptr addrspace(1) %8, align 256, !invariant.load !2
  %16 = load <2 x double>, ptr addrspace(1) %9, align 16, !invariant.load !2
  %.unpack8 = extractelement <2 x double> %16, i32 0
  %.unpack29 = extractelement <2 x double> %16, i32 1
  %17 = load <2 x double>, ptr addrspace(1) %10, align 256, !invariant.load !2
  %.unpack310 = extractelement <2 x double> %17, i32 0
  %.unpack511 = extractelement <2 x double> %17, i32 1
  %18 = trunc i8 %15 to i1
  %19 = zext nneg i32 %14 to i64
  %20 = getelementptr inbounds [16 x i8], ptr addrspace(1) %11, i64 %19
  %.elt = select i1 %18, double %.unpack8, double %.unpack310
  %.elt7 = select i1 %18, double %.unpack29, double %.unpack511
  %21 = insertelement <2 x double> poison, double %.elt, i32 0
  %22 = insertelement <2 x double> %21, double %.elt7, i32 1
  store <2 x double> %22, ptr addrspace(1) %20, align 16
  ret void
}

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.smax.i32(i32, i32) #1

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.umin.i32(i32, i32) #2

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) "nvvm.reqntid"="1,1,1" }
attributes #1 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!2 = !{}
