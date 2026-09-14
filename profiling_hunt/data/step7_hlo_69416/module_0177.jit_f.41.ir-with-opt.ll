; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @loop_concatenate_fusion(ptr noalias nofree readonly align 256 captures(none) dereferenceable(8) %0, ptr noalias nofree readonly align 256 captures(none) dereferenceable(8) %1, ptr noalias nofree readonly align 256 captures(none) dereferenceable(80) %2, ptr noalias nofree readonly align 256 captures(none) dereferenceable(8) %3, ptr noalias nofree readonly align 256 captures(none) dereferenceable(8) %4, ptr noalias nofree writeonly align 256 captures(none) dereferenceable(56) %5) local_unnamed_addr #0 {
  %7 = addrspacecast ptr %0 to ptr addrspace(1)
  %8 = addrspacecast ptr %1 to ptr addrspace(1)
  %9 = addrspacecast ptr %3 to ptr addrspace(1)
  %10 = addrspacecast ptr %4 to ptr addrspace(1)
  %11 = addrspacecast ptr %2 to ptr addrspace(1)
  %12 = addrspacecast ptr %5 to ptr addrspace(1)
  %13 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %14 = icmp samesign ult i32 %13, 3
  br i1 %14, label %15, label %53

15:                                               ; preds = %6
  %.val4 = load double, ptr addrspace(1) %7, align 256, !invariant.load !3
  %.val5 = load double, ptr addrspace(1) %8, align 256, !invariant.load !3
  %.val6 = load i64, ptr addrspace(1) %9, align 256, !invariant.load !3
  %.val7 = load i64, ptr addrspace(1) %10, align 256, !invariant.load !3
  %16 = add i64 %.val7, %.val6
  %17 = icmp slt i64 %16, 0
  %18 = add i64 %16, 10
  %19 = select i1 %17, i64 %18, i64 %16
  %20 = tail call i64 @llvm.smax.i64(i64 %19, i64 0)
  %21 = tail call i64 @llvm.umin.i64(i64 %20, i64 9)
  %22 = getelementptr inbounds [8 x i8], ptr addrspace(1) %11, i64 %21
  %23 = load double, ptr addrspace(1) %22, align 8, !invariant.load !3
  %24 = fsub double %23, %.val4
  %25 = fcmp oeq double %.val4, %.val5
  %26 = fsub double %.val5, %.val4
  %27 = select i1 %25, double 0.000000e+00, double %24
  %28 = select i1 %25, double 1.000000e+00, double %26
  %29 = fdiv double %27, %28
  switch i32 %13, label %39 [
    i32 0, label %30
    i32 2, label %46
  ]

30:                                               ; preds = %15
  %31 = fmul double %29, f0x3FF6FB818F28C58D
  %32 = fmul double %29, f0xBFF0D9735318A381
  %33 = fadd double %29, f0xBFF547A290C89901
  %34 = fmul double %29, %29
  %35 = fsub double %34, %31
  %36 = fmul double %32, %33
  %37 = fadd double %35, f0x3FE6D8F021FA6FA2
  %38 = fmul double %36, %37
  br label %98

39:                                               ; preds = %15
  %40 = fmul double %29, f0x400192C0D16E8162
  %41 = fmul double %29, %29
  %42 = fsub double %41, %40
  %43 = fmul double %41, 1.017000e-01
  %44 = fadd double %42, f0x3FF4B842744B8427
  %45 = fmul double %43, %44
  br label %98

46:                                               ; preds = %15
  %47 = fmul double %29, f0x40031535C3D29ED4
  %48 = fmul double %29, %29
  %49 = fsub double %48, %47
  %50 = fmul double %48, f0x4003ECCDFF932DD6
  %51 = fadd double %49, f0x3FF93FA14B77DC7C
  %52 = fmul double %50, %51
  br label %98

53:                                               ; preds = %6
  %54 = icmp samesign ult i32 %13, 5
  %.val12 = load double, ptr addrspace(1) %7, align 256, !invariant.load !3
  %.val13 = load double, ptr addrspace(1) %8, align 256, !invariant.load !3
  %.val14 = load i64, ptr addrspace(1) %9, align 256, !invariant.load !3
  %.val15 = load i64, ptr addrspace(1) %10, align 256, !invariant.load !3
  %55 = add i64 %.val15, %.val14
  %56 = icmp slt i64 %55, 0
  %57 = add i64 %55, 10
  %58 = select i1 %56, i64 %57, i64 %55
  %59 = tail call i64 @llvm.smax.i64(i64 %58, i64 0)
  %60 = tail call i64 @llvm.umin.i64(i64 %59, i64 9)
  %61 = getelementptr inbounds [8 x i8], ptr addrspace(1) %11, i64 %60
  %62 = load double, ptr addrspace(1) %61, align 8, !invariant.load !3
  %63 = fsub double %62, %.val12
  %64 = fcmp oeq double %.val12, %.val13
  %65 = fsub double %.val13, %.val12
  %66 = select i1 %64, double 0.000000e+00, double %63
  %67 = select i1 %64, double 1.000000e+00, double %65
  %68 = fdiv double %66, %67
  %69 = fmul double %68, %68
  br i1 %54, label %70, label %84

70:                                               ; preds = %53
  %71 = icmp eq i32 %13, 3
  br i1 %71, label %72, label %78

72:                                               ; preds = %70
  %73 = fadd double %68, f0xBFF3795C8B658EFE
  %74 = fmul double %73, f0xC0308C5078902664
  %75 = fadd double %68, f0xBFE3B7F193DF8CAF
  %76 = fmul double %75, %74
  %77 = fmul double %69, %76
  br label %98

78:                                               ; preds = %70
  %79 = fadd double %68, f0xBFF33FC7986B7EF9
  %80 = fmul double %79, f0x4047B0942CF93616
  %81 = fadd double %68, f0xBFE50EB9322562EF
  %82 = fmul double %81, %80
  %83 = fmul double %69, %82
  br label %98

84:                                               ; preds = %53
  %85 = icmp eq i32 %13, 5
  br i1 %85, label %86, label %92

86:                                               ; preds = %84
  %87 = fadd double %68, -1.200000e+00
  %88 = fmul double %87, f0xC0416F71B780910E
  %89 = fadd double %68, f0xBFE5555555555555
  %90 = fmul double %89, %88
  %91 = fmul double %69, %90
  br label %98

92:                                               ; preds = %84
  %93 = fadd double %68, -1.000000e+00
  %94 = fmul double %93, 2.500000e+00
  %95 = fadd double %68, -6.000000e-01
  %96 = fmul double %95, %94
  %97 = fmul double %69, %96
  br label %98

98:                                               ; preds = %30, %46, %39, %78, %72, %92, %86
  %99 = phi double [ %45, %39 ], [ %38, %30 ], [ %52, %46 ], [ %77, %72 ], [ %83, %78 ], [ %97, %92 ], [ %91, %86 ]
  %100 = zext nneg i32 %13 to i64
  %101 = getelementptr inbounds [8 x i8], ptr addrspace(1) %12, i64 %100
  store double %99, ptr addrspace(1) %101, align 8
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.smax.i64(i64, i64) #2

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.umin.i64(i64, i64) #3

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) "nvvm.reqntid"="7,1,1" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!2 = !{i32 0, i32 7}
!3 = !{}
