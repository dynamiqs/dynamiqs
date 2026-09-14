; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @loop_dynamic_update_slice_fusion(ptr noalias nofree align 256 captures(none) dereferenceable(80) %0, ptr noalias nofree readonly align 256 captures(none) dereferenceable(80) %1, ptr noalias nofree readonly align 256 captures(none) dereferenceable(8) %2, ptr noalias nofree readonly align 256 captures(none) dereferenceable(8) %3, ptr noalias nofree readonly align 256 captures(none) dereferenceable(8) %4, ptr noalias nofree readnone align 256 captures(none) dereferenceable(80) %5) local_unnamed_addr #0 {
  %7 = addrspacecast ptr %4 to ptr addrspace(1)
  %8 = addrspacecast ptr %3 to ptr addrspace(1)
  %9 = addrspacecast ptr %2 to ptr addrspace(1)
  %10 = addrspacecast ptr %1 to ptr addrspace(1)
  %11 = addrspacecast ptr %0 to ptr addrspace(1)
  %12 = load i64, ptr addrspace(1) %7, align 256, !invariant.load !2
  %13 = load i64, ptr addrspace(1) %8, align 256, !invariant.load !2
  %14 = add i64 %13, %12
  %15 = icmp slt i64 %14, 0
  %16 = add i64 %14, 10
  %17 = select i1 %15, i64 %16, i64 %14
  %18 = trunc i64 %17 to i32
  %sext = shl i64 %17, 32
  %19 = ashr exact i64 %sext, 32
  %20 = tail call i64 @llvm.smax.i64(i64 %19, i64 0)
  %21 = tail call i64 @llvm.umin.i64(i64 %20, i64 9)
  %22 = icmp ult i32 %18, 10
  %23 = load i64, ptr addrspace(1) %9, align 256, !invariant.load !2
  %24 = add i64 %23, %13
  %25 = icmp slt i64 %24, 0
  %26 = add i64 %24, 10
  %27 = select i1 %25, i64 %26, i64 %24
  %28 = tail call i64 @llvm.smax.i64(i64 %27, i64 0)
  %29 = tail call i64 @llvm.umin.i64(i64 %28, i64 9)
  %30 = getelementptr inbounds [8 x i8], ptr addrspace(1) %10, i64 %29
  %31 = load double, ptr addrspace(1) %30, align 8, !invariant.load !2
  %32 = getelementptr inbounds [8 x i8], ptr addrspace(1) %11, i64 %21
  %33 = load double, ptr addrspace(1) %32, align 8
  %34 = select i1 %22, double %31, double %33
  store double %34, ptr addrspace(1) %32, align 8
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
