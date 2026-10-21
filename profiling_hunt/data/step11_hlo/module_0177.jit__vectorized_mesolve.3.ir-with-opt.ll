; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @loop_minimum_multiply_select_fusion(ptr noalias nofree readonly align 16 captures(none) dereferenceable(320) %0, ptr noalias nofree writeonly align 256 captures(none) dereferenceable(8) initializes((0, 8)) %1, ptr noalias nofree writeonly align 256 captures(none) dereferenceable(8) initializes((0, 8)) %2, ptr noalias nofree writeonly align 256 captures(none) dereferenceable(8) initializes((0, 8)) %3, ptr noalias nofree writeonly align 256 captures(none) dereferenceable(8) initializes((0, 8)) %4) local_unnamed_addr #0 {
  %6 = addrspacecast ptr %0 to ptr addrspace(1)
  %7 = addrspacecast ptr %1 to ptr addrspace(1)
  %8 = addrspacecast ptr %2 to ptr addrspace(1)
  %9 = addrspacecast ptr %3 to ptr addrspace(1)
  %10 = addrspacecast ptr %4 to ptr addrspace(1)
  %11 = load double, ptr addrspace(1) %6, align 16, !invariant.load !2
  %12 = getelementptr inbounds i8, ptr addrspace(1) %6, i64 312
  %13 = load double, ptr addrspace(1) %12, align 8, !invariant.load !2
  %14 = fcmp olt double %11, %13
  %15 = select i1 %14, i64 1, i64 -1
  %16 = sitofp i64 %15 to double
  %17 = fmul double %11, %16
  %18 = fadd double %17, 1.000000e-02
  %19 = fmul double %13, %16
  %20 = tail call double @llvm.minimum.f64(double %18, double %19)
  store double %20, ptr addrspace(1) %7, align 256
  store double %19, ptr addrspace(1) %8, align 256
  store double %17, ptr addrspace(1) %9, align 256
  store i64 %15, ptr addrspace(1) %10, align 256
  ret void
}

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.minimum.f64(double, double) #1

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) "nvvm.reqntid"="1,1,1" }
attributes #1 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!2 = !{}
