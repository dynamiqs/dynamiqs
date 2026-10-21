; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @wrapped_dynamic_slice(ptr noalias nofree readonly align 16 captures(none) dereferenceable(2048) %0, ptr noalias nofree readonly align 16 captures(none) dereferenceable(8) %1, ptr noalias nofree readonly align 16 captures(none) dereferenceable(8) %2, ptr noalias nofree writeonly align 256 captures(none) dereferenceable(32) %3) local_unnamed_addr #0 {
  %5 = addrspacecast ptr %1 to ptr addrspace(1)
  %6 = addrspacecast ptr %2 to ptr addrspace(1)
  %7 = addrspacecast ptr %0 to ptr addrspace(1)
  %8 = addrspacecast ptr %3 to ptr addrspace(1)
  %9 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %10 = zext nneg i32 %9 to i64
  %11 = load i64, ptr addrspace(1) %5, align 16, !invariant.load !3
  %12 = icmp sgt i64 %11, 0
  %13 = load i64, ptr addrspace(1) %6, align 16, !invariant.load !3
  %14 = tail call i64 @llvm.smax.i64(i64 %13, i64 0)
  %15 = tail call i64 @llvm.umin.i64(i64 %14, i64 62)
  %16 = select i1 %12, i64 64, i64 0
  %17 = getelementptr inbounds [16 x i8], ptr addrspace(1) %7, i64 %16
  %18 = getelementptr inbounds [16 x i8], ptr addrspace(1) %17, i64 %15
  %19 = getelementptr inbounds [16 x i8], ptr addrspace(1) %18, i64 %10
  %20 = load <2 x double>, ptr addrspace(1) %19, align 16, !invariant.load !3
  %.unpack5 = extractelement <2 x double> %20, i32 0
  %.unpack26 = extractelement <2 x double> %20, i32 1
  %21 = getelementptr inbounds [16 x i8], ptr addrspace(1) %8, i64 %10
  %22 = insertelement <2 x double> poison, double %.unpack5, i32 0
  %23 = insertelement <2 x double> %22, double %.unpack26, i32 1
  store <2 x double> %23, ptr addrspace(1) %21, align 16
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.smax.i64(i64, i64) #2

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.umin.i64(i64, i64) #3

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) "nvvm.reqntid"="2,1,1" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!2 = !{i32 0, i32 2}
!3 = !{}
