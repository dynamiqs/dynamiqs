; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @loop_and_fusion_1(ptr noalias nofree readonly align 256 captures(none) dereferenceable(1) %0, ptr noalias nofree readonly align 256 captures(none) dereferenceable(8) %1, ptr noalias nofree readonly align 256 captures(none) dereferenceable(320) %2, ptr noalias nofree readonly align 256 captures(none) dereferenceable(8) %3, ptr noalias nofree readonly align 256 captures(none) dereferenceable(8) %4, ptr noalias nofree writeonly align 256 captures(none) dereferenceable(1) initializes((0, 1)) %5) local_unnamed_addr #0 {
  %7 = addrspacecast ptr %4 to ptr addrspace(1)
  %8 = addrspacecast ptr %3 to ptr addrspace(1)
  %9 = addrspacecast ptr %2 to ptr addrspace(1)
  %10 = addrspacecast ptr %1 to ptr addrspace(1)
  %11 = addrspacecast ptr %0 to ptr addrspace(1)
  %12 = addrspacecast ptr %5 to ptr addrspace(1)
  %13 = load i64, ptr addrspace(1) %7, align 256, !invariant.load !2
  %14 = load i64, ptr addrspace(1) %8, align 256, !invariant.load !2
  %15 = add i64 %14, %13
  %16 = icmp slt i64 %15, 0
  %17 = add i64 %15, 40
  %18 = select i1 %16, i64 %17, i64 %15
  %19 = tail call i64 @llvm.smax.i64(i64 %18, i64 0)
  %20 = tail call i64 @llvm.umin.i64(i64 %19, i64 39)
  %21 = getelementptr inbounds [8 x i8], ptr addrspace(1) %9, i64 %20
  %22 = load double, ptr addrspace(1) %21, align 8, !invariant.load !2
  %23 = load double, ptr addrspace(1) %10, align 256, !invariant.load !2
  %24 = load i8, ptr addrspace(1) %11, align 256, !invariant.load !2
  %25 = fcmp ole double %22, %23
  %26 = icmp slt i64 %15, 40
  %27 = icmp slt i64 %14, 40
  %28 = and i8 %24, 1
  %29 = and i1 %27, %26
  %30 = select i1 %29, i1 %25, i1 false
  %31 = select i1 %30, i8 %28, i8 0
  store i8 %31, ptr addrspace(1) %12, align 256
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
