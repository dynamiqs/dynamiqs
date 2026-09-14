; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @loop_multiply_fusion(ptr noalias nofree readonly align 256 captures(none) dereferenceable(32768) %0, ptr noalias nofree align 256 captures(none) dereferenceable(65536) %1, ptr noalias nofree readonly align 256 captures(none) dereferenceable(65536) %2, ptr noalias nofree readonly align 256 captures(none) dereferenceable(65536) %3, ptr noalias nofree readonly align 256 captures(none) dereferenceable(1) %4, ptr noalias nofree readnone align 256 captures(none) dereferenceable(65536) %5) local_unnamed_addr #0 {
  %7 = addrspacecast ptr %4 to ptr addrspace(1)
  %8 = addrspacecast ptr %1 to ptr addrspace(1)
  %9 = addrspacecast ptr %2 to ptr addrspace(1)
  %10 = addrspacecast ptr %0 to ptr addrspace(1)
  %11 = addrspacecast ptr %3 to ptr addrspace(1)
  %12 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !4
  %13 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !5
  %14 = load i8, ptr addrspace(1) %7, align 256, !invariant.load !6
  %15 = shl nuw nsw i32 %12, 7
  %16 = or disjoint i32 %15, %13
  %17 = zext nneg i32 %16 to i64
  %18 = getelementptr inbounds [16 x i8], ptr addrspace(1) %8, i64 %17
  %19 = load <2 x double>, ptr addrspace(1) %18, align 16
  %.unpack11 = extractelement <2 x double> %19, i32 0
  %.unpack212 = extractelement <2 x double> %19, i32 1
  %20 = getelementptr inbounds [16 x i8], ptr addrspace(1) %9, i64 %17
  %21 = load <2 x double>, ptr addrspace(1) %20, align 16, !invariant.load !6
  %.unpack313 = extractelement <2 x double> %21, i32 0
  %.unpack514 = extractelement <2 x double> %21, i32 1
  %22 = trunc i8 %14 to i1
  %23 = getelementptr inbounds [8 x i8], ptr addrspace(1) %10, i64 %17
  %24 = load double, ptr addrspace(1) %23, align 8, !invariant.load !6
  %25 = select i1 %22, double %.unpack11, double %.unpack313
  %26 = select i1 %22, double %.unpack212, double %.unpack514
  %27 = tail call double @llvm.nvvm.fabs.f64(double %25)
  %28 = tail call double @llvm.nvvm.fabs.f64(double %26)
  %29 = tail call double @llvm.maximum.f64(double %27, double %28)
  %30 = tail call double @llvm.minimum.f64(double %27, double %28)
  %31 = fdiv double %30, %29
  %32 = fmul double %31, %31
  %33 = fadd double %32, 1.000000e+00
  %34 = tail call double @llvm.sqrt.f64(double %33)
  %35 = fmul double %29, %34
  %36 = fcmp uno double %35, 0.000000e+00
  %37 = select i1 %36, double %30, double %35
  %38 = tail call double @llvm.maximum.f64(double %24, double %37)
  %39 = fmul double %38, f0x3EB0C6F7A0B5ED8D
  %40 = getelementptr inbounds [16 x i8], ptr addrspace(1) %11, i64 %17
  %41 = load <2 x double>, ptr addrspace(1) %40, align 16, !invariant.load !6
  %.unpack615 = extractelement <2 x double> %41, i32 0
  %.unpack816 = extractelement <2 x double> %41, i32 1
  %42 = fcmp uno double %.unpack615, %.unpack816
  %43 = fadd double %39, f0x3EB0C6F7A0B5ED8D
  %44 = select i1 %42, double +inf, double %.unpack615
  %45 = select i1 %42, double 0.000000e+00, double %.unpack816
  %46 = fdiv double %43, 0.000000e+00
  %47 = fmul double %43, %46
  %48 = fmul double %44, %46
  %49 = fadd double %45, %48
  %50 = fdiv double %49, %47
  %51 = fmul double %45, %46
  %52 = fsub double %51, %44
  %53 = fdiv double %52, %47
  %54 = fdiv double 0.000000e+00, %43
  %55 = fmul double %54, 0.000000e+00
  %56 = fadd double %43, %55
  %57 = fmul double %45, %54
  %58 = fadd double %44, %57
  %59 = fdiv double %58, %56
  %60 = fmul double %44, %54
  %61 = fsub double %45, %60
  %62 = fdiv double %61, %56
  %63 = tail call double @llvm.nvvm.fabs.f64(double %43)
  %64 = fcmp oeq double %63, 0.000000e+00
  %65 = fcmp ord double %44, 0.000000e+00
  %66 = fcmp ord double %45, 0.000000e+00
  %67 = or i1 %65, %66
  %68 = and i1 %67, %64
  %69 = tail call double @llvm.copysign.f64(double +inf, double %43)
  %70 = fmul double %44, %69
  %71 = fmul double %45, %69
  %72 = fcmp one double %63, +inf
  %73 = tail call double @llvm.nvvm.fabs.f64(double %44)
  %74 = fcmp oeq double %73, +inf
  %75 = tail call double @llvm.nvvm.fabs.f64(double %45)
  %76 = fcmp oeq double %75, +inf
  %77 = or i1 %74, %76
  %78 = and i1 %77, %72
  %79 = select i1 %74, double 1.000000e+00, double 0.000000e+00
  %80 = tail call double @llvm.copysign.f64(double %79, double %44)
  %81 = select i1 %76, double 1.000000e+00, double 0.000000e+00
  %82 = tail call double @llvm.copysign.f64(double %81, double %45)
  %83 = fmul double %80, %43
  %84 = tail call double @llvm.copysign.f64(double 0.000000e+00, double %45)
  %85 = fadd double %84, %83
  %86 = fmul double %85, +inf
  %87 = tail call double @llvm.copysign.f64(double 0.000000e+00, double %44)
  %88 = fmul double %82, %43
  %89 = fsub double %88, %87
  %90 = fmul double %89, +inf
  %91 = fcmp one double %73, +inf
  %92 = fcmp one double %75, +inf
  %93 = and i1 %91, %92
  %94 = fcmp oeq double %63, +inf
  %95 = and i1 %93, %94
  %96 = select i1 %94, double 1.000000e+00, double 0.000000e+00
  %97 = tail call double @llvm.copysign.f64(double %96, double %43)
  %98 = fmul double %44, %97
  %99 = fmul double %45, 0.000000e+00
  %100 = fadd double %99, %98
  %101 = fmul double %100, 0.000000e+00
  %102 = fmul double %45, %97
  %103 = fmul double %44, 0.000000e+00
  %104 = fsub double %102, %103
  %105 = fmul double %104, 0.000000e+00
  %106 = fcmp olt double %63, 0.000000e+00
  %107 = select i1 %106, double %50, double %59
  %108 = select i1 %106, double %53, double %62
  %109 = select i1 %95, double %101, double %107
  %110 = select i1 %95, double %105, double %108
  %111 = select i1 %78, double %86, double %109
  %112 = select i1 %78, double %90, double %110
  %113 = select i1 %68, double %70, double %111
  %114 = select i1 %68, double %71, double %112
  %115 = fcmp uno double %107, 0.000000e+00
  %116 = fcmp uno double %108, 0.000000e+00
  %117 = and i1 %115, %116
  %118 = select i1 %117, double %113, double %107
  %119 = select i1 %117, double %114, double %108
  %120 = fmul double %118, %118
  %121 = fmul double %118, %119
  %122 = fmul double %119, %119
  %123 = fadd double %120, %122
  %124 = fsub double %121, %121
  %125 = insertelement <2 x double> poison, double %123, i32 0
  %126 = insertelement <2 x double> %125, double %124, i32 1
  store <2 x double> %126, ptr addrspace(1) %18, align 16
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
