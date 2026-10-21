; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @loop_multiply_fusion_1(ptr noalias nofree readonly align 256 captures(none) dereferenceable(32768) %0, ptr noalias nofree align 256 captures(none) dereferenceable(65536) %1, ptr noalias nofree readonly align 256 captures(none) dereferenceable(65536) %2, ptr noalias nofree readonly align 256 captures(none) dereferenceable(65536) %3, ptr noalias nofree readonly align 16 captures(none) dereferenceable(8) %4, ptr noalias nofree readonly align 16 captures(none) dereferenceable(8) %5, ptr noalias nofree readonly align 256 captures(none) dereferenceable(1) %6, ptr noalias nofree readnone align 256 captures(none) dereferenceable(65536) %7) local_unnamed_addr #0 {
  %9 = addrspacecast ptr %6 to ptr addrspace(1)
  %10 = addrspacecast ptr %1 to ptr addrspace(1)
  %11 = addrspacecast ptr %2 to ptr addrspace(1)
  %12 = addrspacecast ptr %0 to ptr addrspace(1)
  %13 = addrspacecast ptr %4 to ptr addrspace(1)
  %14 = addrspacecast ptr %5 to ptr addrspace(1)
  %15 = addrspacecast ptr %3 to ptr addrspace(1)
  %16 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !4
  %17 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !5
  %18 = load i8, ptr addrspace(1) %9, align 256, !invariant.load !6
  %19 = shl nuw nsw i32 %16, 7
  %20 = or disjoint i32 %19, %17
  %21 = zext nneg i32 %20 to i64
  %22 = getelementptr inbounds [16 x i8], ptr addrspace(1) %10, i64 %21
  %23 = load <2 x double>, ptr addrspace(1) %22, align 16
  %.unpack11 = extractelement <2 x double> %23, i32 0
  %.unpack212 = extractelement <2 x double> %23, i32 1
  %24 = getelementptr inbounds [16 x i8], ptr addrspace(1) %11, i64 %21
  %25 = load <2 x double>, ptr addrspace(1) %24, align 16, !invariant.load !6
  %.unpack313 = extractelement <2 x double> %25, i32 0
  %.unpack514 = extractelement <2 x double> %25, i32 1
  %26 = trunc i8 %18 to i1
  %27 = getelementptr inbounds [8 x i8], ptr addrspace(1) %12, i64 %21
  %28 = load double, ptr addrspace(1) %27, align 8, !invariant.load !6
  %29 = select i1 %26, double %.unpack11, double %.unpack313
  %30 = select i1 %26, double %.unpack212, double %.unpack514
  %31 = tail call double @llvm.nvvm.fabs.f64(double %29)
  %32 = tail call double @llvm.nvvm.fabs.f64(double %30)
  %33 = tail call double @llvm.maximum.f64(double %31, double %32)
  %34 = tail call double @llvm.minimum.f64(double %31, double %32)
  %35 = fdiv double %34, %33
  %36 = fmul double %35, %35
  %37 = fadd double %36, 1.000000e+00
  %38 = tail call double @llvm.sqrt.f64(double %37)
  %39 = fmul double %33, %38
  %40 = fcmp uno double %39, 0.000000e+00
  %41 = select i1 %40, double %34, double %39
  %42 = tail call double @llvm.maximum.f64(double %28, double %41)
  %43 = load double, ptr addrspace(1) %13, align 16, !invariant.load !6
  %44 = load double, ptr addrspace(1) %14, align 16, !invariant.load !6
  %45 = fmul double %43, %42
  %46 = getelementptr inbounds [16 x i8], ptr addrspace(1) %15, i64 %21
  %47 = load <2 x double>, ptr addrspace(1) %46, align 16, !invariant.load !6
  %.unpack615 = extractelement <2 x double> %47, i32 0
  %.unpack816 = extractelement <2 x double> %47, i32 1
  %48 = fcmp uno double %.unpack615, %.unpack816
  %49 = fadd double %44, %45
  %50 = select i1 %48, double +inf, double %.unpack615
  %51 = select i1 %48, double 0.000000e+00, double %.unpack816
  %52 = fdiv double %49, 0.000000e+00
  %53 = fmul double %49, %52
  %54 = fmul double %50, %52
  %55 = fadd double %51, %54
  %56 = fdiv double %55, %53
  %57 = fmul double %51, %52
  %58 = fsub double %57, %50
  %59 = fdiv double %58, %53
  %60 = fdiv double 0.000000e+00, %49
  %61 = fmul double %60, 0.000000e+00
  %62 = fadd double %49, %61
  %63 = fmul double %51, %60
  %64 = fadd double %50, %63
  %65 = fdiv double %64, %62
  %66 = fmul double %50, %60
  %67 = fsub double %51, %66
  %68 = fdiv double %67, %62
  %69 = tail call double @llvm.nvvm.fabs.f64(double %49)
  %70 = fcmp oeq double %69, 0.000000e+00
  %71 = fcmp ord double %50, 0.000000e+00
  %72 = fcmp ord double %51, 0.000000e+00
  %73 = or i1 %71, %72
  %74 = and i1 %73, %70
  %75 = tail call double @llvm.copysign.f64(double +inf, double %49)
  %76 = fmul double %50, %75
  %77 = fmul double %51, %75
  %78 = fcmp one double %69, +inf
  %79 = tail call double @llvm.nvvm.fabs.f64(double %50)
  %80 = fcmp oeq double %79, +inf
  %81 = tail call double @llvm.nvvm.fabs.f64(double %51)
  %82 = fcmp oeq double %81, +inf
  %83 = or i1 %80, %82
  %84 = and i1 %83, %78
  %85 = select i1 %80, double 1.000000e+00, double 0.000000e+00
  %86 = tail call double @llvm.copysign.f64(double %85, double %50)
  %87 = select i1 %82, double 1.000000e+00, double 0.000000e+00
  %88 = tail call double @llvm.copysign.f64(double %87, double %51)
  %89 = fmul double %86, %49
  %90 = tail call double @llvm.copysign.f64(double 0.000000e+00, double %51)
  %91 = fadd double %90, %89
  %92 = fmul double %91, +inf
  %93 = tail call double @llvm.copysign.f64(double 0.000000e+00, double %50)
  %94 = fmul double %88, %49
  %95 = fsub double %94, %93
  %96 = fmul double %95, +inf
  %97 = fcmp one double %79, +inf
  %98 = fcmp one double %81, +inf
  %99 = and i1 %97, %98
  %100 = fcmp oeq double %69, +inf
  %101 = and i1 %99, %100
  %102 = select i1 %100, double 1.000000e+00, double 0.000000e+00
  %103 = tail call double @llvm.copysign.f64(double %102, double %49)
  %104 = fmul double %50, %103
  %105 = fmul double %51, 0.000000e+00
  %106 = fadd double %105, %104
  %107 = fmul double %106, 0.000000e+00
  %108 = fmul double %51, %103
  %109 = fmul double %50, 0.000000e+00
  %110 = fsub double %108, %109
  %111 = fmul double %110, 0.000000e+00
  %112 = fcmp olt double %69, 0.000000e+00
  %113 = select i1 %112, double %56, double %65
  %114 = select i1 %112, double %59, double %68
  %115 = select i1 %101, double %107, double %113
  %116 = select i1 %101, double %111, double %114
  %117 = select i1 %84, double %92, double %115
  %118 = select i1 %84, double %96, double %116
  %119 = select i1 %74, double %76, double %117
  %120 = select i1 %74, double %77, double %118
  %121 = fcmp uno double %113, 0.000000e+00
  %122 = fcmp uno double %114, 0.000000e+00
  %123 = and i1 %121, %122
  %124 = select i1 %123, double %119, double %113
  %125 = select i1 %123, double %120, double %114
  %126 = fmul double %124, %124
  %127 = fmul double %124, %125
  %128 = fmul double %125, %125
  %129 = fadd double %126, %128
  %130 = fsub double %127, %127
  %131 = insertelement <2 x double> poison, double %129, i32 0
  %132 = insertelement <2 x double> %131, double %130, i32 1
  store <2 x double> %132, ptr addrspace(1) %22, align 16
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.maximum.f64(double, double) #2

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.minimum.f64(double, double) #2

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.nvvm.fabs.f64(double) #2

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.copysign.f64(double, double) #2

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.sqrt.f64(double) #3

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) "nvvm.reqntid"="128,1,1" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}
!nvvmir.version = !{!3}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!2 = !{!"clang version 3.8.0 (tags/RELEASE_380/final)"}
!3 = !{i32 2, i32 0}
!4 = !{i32 0, i32 32}
!5 = !{i32 0, i32 128}
!6 = !{}
