; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @loop_complex_fusion(ptr noalias nofree readonly align 16 captures(none) dereferenceable(1008) %0, ptr noalias nofree writeonly align 256 captures(none) dereferenceable(1008) %1) local_unnamed_addr #0 {
  %3 = addrspacecast ptr %0 to ptr addrspace(1)
  %4 = addrspacecast ptr %1 to ptr addrspace(1)
  %5 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !4
  %6 = zext nneg i32 %5 to i64
  %7 = getelementptr inbounds [16 x i8], ptr addrspace(1) %3, i64 %6
  %8 = load <2 x double>, ptr addrspace(1) %7, align 16, !invariant.load !5
  %.unpack5 = extractelement <2 x double> %8, i32 0
  %.unpack26 = extractelement <2 x double> %8, i32 1
  %9 = tail call double @llvm.nvvm.fabs.f64(double %.unpack5)
  %10 = tail call double @llvm.nvvm.fabs.f64(double %.unpack26)
  %11 = fcmp olt double %9, %10
  %12 = fdiv double %9, %10
  %13 = fdiv double %10, %9
  %14 = fcmp oeq double %9, %10
  %15 = select i1 %11, double %12, double %13
  %16 = select i1 %14, double 1.000000e+00, double %15
  %17 = tail call double @llvm.nvvm.fabs.f64(double %16)
  %18 = tail call double @llvm.minimum.f64(double %17, double 1.000000e+00)
  %19 = tail call double @llvm.maximum.f64(double %17, double 1.000000e+00)
  %20 = fdiv double %18, %19
  %21 = fmul double %20, %20
  %22 = fmul double %19, %21
  %23 = fadd double %21, 1.000000e+00
  %24 = tail call double @llvm.minimum.f64(double %9, double %10)
  %25 = tail call double @llvm.maximum.f64(double %9, double %10)
  %26 = fdiv double %24, %25
  %27 = tail call double @llvm.sqrt.f64(double %23)
  %28 = fcmp oeq double %27, 1.000000e+00
  %29 = fcmp ogt double %21, 0.000000e+00
  %30 = fmul double %22, 5.000000e-01
  %31 = fmul double %26, %26
  %32 = fmul double %25, %31
  %33 = fadd double %31, 1.000000e+00
  %34 = and i1 %29, %28
  %35 = fadd double %19, %30
  %36 = fmul double %19, %27
  %37 = tail call double @llvm.sqrt.f64(double %33)
  %38 = fcmp oeq double %37, 1.000000e+00
  %39 = fcmp ogt double %31, 0.000000e+00
  %40 = fmul double %32, 5.000000e-01
  %41 = fcmp oeq double %19, %18
  %42 = fmul double %19, f0x3FF6A09E667F3BCD
  %43 = select i1 %34, double %35, double %36
  %44 = and i1 %39, %38
  %45 = fadd double %25, %40
  %46 = fmul double %25, %37
  %47 = tail call double @llvm.sqrt.f64(double %9)
  %48 = tail call double @llvm.sqrt.f64(double %10)
  %49 = fdiv double %47, %48
  %50 = fdiv double %48, %47
  %51 = select i1 %41, double %42, double %43
  %52 = fcmp oeq double %25, %24
  %53 = fmul double %25, f0x3FF6A09E667F3BCD
  %54 = select i1 %44, double %45, double %46
  %55 = select i1 %11, double %49, double %50
  %56 = fadd double %51, 1.000000e+00
  %57 = fadd double %16, %51
  %58 = select i1 %52, double %53, double %54
  %59 = select i1 %14, double 1.000000e+00, double %55
  %60 = tail call double @llvm.sqrt.f64(double %56)
  %61 = tail call double @llvm.sqrt.f64(double %57)
  %62 = fmul double %60, f0x3FE6A09E667F3BCC
  %63 = fmul double %61, f0x3FE6A09E667F3BCC
  %64 = fadd double %9, %58
  %65 = fmul double %48, %59
  %66 = fmul double %60, f0x3FF6A09E667F3BCD
  %67 = fmul double %61, f0x3FF6A09E667F3BCD
  %68 = fmul double %47, %62
  %69 = fmul double %48, %63
  %70 = fmul double %64, 5.000000e-01
  %71 = tail call double @llvm.sqrt.f64(double %70)
  %72 = fcmp ogt double %9, %10
  %73 = fdiv double %65, %66
  %74 = fdiv double %48, %67
  %75 = fmul double %71, 2.000000e+00
  %76 = select i1 %72, double %68, double %69
  %77 = tail call i1 @llvm.is.fpclass.f64(double %71, i32 608)
  %78 = select i1 %72, double %73, double %74
  %79 = fdiv double %10, %75
  %80 = fmul double %47, f0x3FF19435CAFFA9F8
  %81 = select i1 %77, double %76, double %71
  %82 = fmul double %48, f0x3FDD203138F6C828
  %83 = select i1 %77, double %78, double %79
  %84 = select i1 %14, double %80, double %81
  %85 = fneg double %84
  %86 = fcmp olt double %.unpack26, 0.000000e+00
  %87 = select i1 %14, double %82, double %83
  %88 = fneg double %87
  %89 = fcmp oge double %.unpack5, 0.000000e+00
  %90 = fcmp olt double %.unpack5, 0.000000e+00
  %91 = select i1 %86, double %85, double %84
  %92 = select i1 %86, double %88, double %87
  %93 = select i1 %89, double %84, double %87
  %94 = select i1 %90, double %91, double %92
  %95 = getelementptr inbounds [16 x i8], ptr addrspace(1) %4, i64 %6
  %96 = insertelement <2 x double> poison, double %93, i32 0
  %97 = insertelement <2 x double> %96, double %94, i32 1
  store <2 x double> %97, ptr addrspace(1) %95, align 16
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.minimum.f64(double, double) #2

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.maximum.f64(double, double) #2

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.nvvm.fabs.f64(double) #2

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i1 @llvm.is.fpclass.f64(double, i32 immarg) #3

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.sqrt.f64(double) #3

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) "nvvm.reqntid"="63,1,1" }
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
!4 = !{i32 0, i32 63}
!5 = !{}
