; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @loop_dynamic_slice_fusion(ptr noalias nofree readonly align 16 captures(none) dereferenceable(1024) %0, ptr noalias nofree readonly align 16 captures(none) dereferenceable(8) %1, ptr noalias nofree writeonly align 256 captures(none) dereferenceable(16) initializes((0, 16)) %2) local_unnamed_addr #0 {
  %4 = addrspacecast ptr %1 to ptr addrspace(1)
  %5 = addrspacecast ptr %0 to ptr addrspace(1)
  %6 = addrspacecast ptr %2 to ptr addrspace(1)
  %7 = load i64, ptr addrspace(1) %4, align 16, !invariant.load !2
  %8 = tail call i64 @llvm.smax.i64(i64 %7, i64 0)
  %9 = tail call i64 @llvm.umin.i64(i64 %8, i64 63)
  %10 = getelementptr inbounds [16 x i8], ptr addrspace(1) %5, i64 %9
  %11 = load <2 x double>, ptr addrspace(1) %10, align 16, !invariant.load !2
  %.unpack5 = extractelement <2 x double> %11, i32 0
  %.unpack26 = extractelement <2 x double> %11, i32 1
  %12 = insertelement <2 x double> poison, double %.unpack5, i32 0
  %13 = insertelement <2 x double> %12, double %.unpack26, i32 1
  store <2 x double> %13, ptr addrspace(1) %6, align 256
  ret void
}

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.smax.i64(i64, i64) #1

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.umin.i64(i64, i64) #2

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) "nvvm.reqntid"="1,1,1" }
attributes #1 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!2 = !{}
