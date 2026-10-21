; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @wrapped_broadcast(ptr noalias nofree readonly align 16 captures(none) dereferenceable(16) %0, ptr noalias nofree writeonly align 256 captures(none) dereferenceable(1024) %1) local_unnamed_addr #0 {
  %3 = addrspacecast ptr %0 to ptr addrspace(1)
  %4 = addrspacecast ptr %1 to ptr addrspace(1)
  %5 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %6 = load <2 x double>, ptr addrspace(1) %3, align 16, !invariant.load !3
  %.unpack5 = extractelement <2 x double> %6, i32 0
  %.unpack26 = extractelement <2 x double> %6, i32 1
  %7 = zext nneg i32 %5 to i64
  %8 = getelementptr inbounds [16 x i8], ptr addrspace(1) %4, i64 %7
  %9 = insertelement <2 x double> poison, double %.unpack5, i32 0
  %10 = insertelement <2 x double> %9, double %.unpack26, i32 1
  store <2 x double> %10, ptr addrspace(1) %8, align 16
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) "nvvm.reqntid"="64,1,1" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!2 = !{i32 0, i32 64}
!3 = !{}
