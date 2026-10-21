; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @loop_reduce_fusion_1(ptr noalias align 256 dereferenceable(458752) %0, ptr noalias align 256 dereferenceable(112) %1, ptr noalias align 256 dereferenceable(65536) %2) #0 {
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !1
  %5 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %6 = getelementptr inbounds [7 x { double, double }], ptr %1, i32 0, i32 0
  %7 = load { double, double }, ptr %6, align 8, !invariant.load !3
  %8 = mul i32 %4, 128
  %9 = add i32 %8, %5
  %10 = getelementptr inbounds [28672 x { double, double }], ptr %0, i32 0, i32 %9
  %11 = load { double, double }, ptr %10, align 8, !invariant.load !3
  %12 = extractvalue { double, double } %7, 0
  %13 = extractvalue { double, double } %7, 1
  %14 = extractvalue { double, double } %11, 0
  %15 = extractvalue { double, double } %11, 1
  %16 = fmul double %12, %14
  %17 = fmul double %13, %15
  %18 = fmul double %13, %14
  %19 = fmul double %12, %15
  %20 = fsub double %16, %17
  %21 = fadd double %18, %19
  %22 = fadd double %20, 0.000000e+00
  %23 = fadd double %21, 0.000000e+00
  %24 = getelementptr inbounds [7 x { double, double }], ptr %1, i32 0, i32 1
  %25 = load { double, double }, ptr %24, align 8, !invariant.load !3
  %26 = add i32 %9, 4096
  %27 = getelementptr inbounds [28672 x { double, double }], ptr %0, i32 0, i32 %26
  %28 = load { double, double }, ptr %27, align 8, !invariant.load !3
  %29 = extractvalue { double, double } %25, 0
  %30 = extractvalue { double, double } %25, 1
  %31 = extractvalue { double, double } %28, 0
  %32 = extractvalue { double, double } %28, 1
  %33 = fmul double %29, %31
  %34 = fmul double %30, %32
  %35 = fmul double %30, %31
  %36 = fmul double %29, %32
  %37 = fsub double %33, %34
  %38 = fadd double %35, %36
  %39 = fadd double %22, %37
  %40 = fadd double %23, %38
  %41 = getelementptr inbounds [7 x { double, double }], ptr %1, i32 0, i32 2
  %42 = load { double, double }, ptr %41, align 8, !invariant.load !3
  %43 = add i32 %9, 8192
  %44 = getelementptr inbounds [28672 x { double, double }], ptr %0, i32 0, i32 %43
  %45 = load { double, double }, ptr %44, align 8, !invariant.load !3
  %46 = extractvalue { double, double } %42, 0
  %47 = extractvalue { double, double } %42, 1
  %48 = extractvalue { double, double } %45, 0
  %49 = extractvalue { double, double } %45, 1
  %50 = fmul double %46, %48
  %51 = fmul double %47, %49
  %52 = fmul double %47, %48
  %53 = fmul double %46, %49
  %54 = fsub double %50, %51
  %55 = fadd double %52, %53
  %56 = fadd double %39, %54
  %57 = fadd double %40, %55
  %58 = getelementptr inbounds [7 x { double, double }], ptr %1, i32 0, i32 3
  %59 = load { double, double }, ptr %58, align 8, !invariant.load !3
  %60 = add i32 %9, 12288
  %61 = getelementptr inbounds [28672 x { double, double }], ptr %0, i32 0, i32 %60
  %62 = load { double, double }, ptr %61, align 8, !invariant.load !3
  %63 = extractvalue { double, double } %59, 0
  %64 = extractvalue { double, double } %59, 1
  %65 = extractvalue { double, double } %62, 0
  %66 = extractvalue { double, double } %62, 1
  %67 = fmul double %63, %65
  %68 = fmul double %64, %66
  %69 = fmul double %64, %65
  %70 = fmul double %63, %66
  %71 = fsub double %67, %68
  %72 = fadd double %69, %70
  %73 = fadd double %56, %71
  %74 = fadd double %57, %72
  %75 = getelementptr inbounds [7 x { double, double }], ptr %1, i32 0, i32 4
  %76 = load { double, double }, ptr %75, align 8, !invariant.load !3
  %77 = add i32 %9, 16384
  %78 = getelementptr inbounds [28672 x { double, double }], ptr %0, i32 0, i32 %77
  %79 = load { double, double }, ptr %78, align 8, !invariant.load !3
  %80 = extractvalue { double, double } %76, 0
  %81 = extractvalue { double, double } %76, 1
  %82 = extractvalue { double, double } %79, 0
  %83 = extractvalue { double, double } %79, 1
  %84 = fmul double %80, %82
  %85 = fmul double %81, %83
  %86 = fmul double %81, %82
  %87 = fmul double %80, %83
  %88 = fsub double %84, %85
  %89 = fadd double %86, %87
  %90 = fadd double %73, %88
  %91 = fadd double %74, %89
  %92 = getelementptr inbounds [7 x { double, double }], ptr %1, i32 0, i32 5
  %93 = load { double, double }, ptr %92, align 8, !invariant.load !3
  %94 = add i32 %9, 20480
  %95 = getelementptr inbounds [28672 x { double, double }], ptr %0, i32 0, i32 %94
  %96 = load { double, double }, ptr %95, align 8, !invariant.load !3
  %97 = extractvalue { double, double } %93, 0
  %98 = extractvalue { double, double } %93, 1
  %99 = extractvalue { double, double } %96, 0
  %100 = extractvalue { double, double } %96, 1
  %101 = fmul double %97, %99
  %102 = fmul double %98, %100
  %103 = fmul double %98, %99
  %104 = fmul double %97, %100
  %105 = fsub double %101, %102
  %106 = fadd double %103, %104
  %107 = fadd double %90, %105
  %108 = fadd double %91, %106
  %109 = getelementptr inbounds [7 x { double, double }], ptr %1, i32 0, i32 6
  %110 = load { double, double }, ptr %109, align 8, !invariant.load !3
  %111 = add i32 %9, 24576
  %112 = getelementptr inbounds [28672 x { double, double }], ptr %0, i32 0, i32 %111
  %113 = load { double, double }, ptr %112, align 8, !invariant.load !3
  %114 = extractvalue { double, double } %110, 0
  %115 = extractvalue { double, double } %110, 1
  %116 = extractvalue { double, double } %113, 0
  %117 = extractvalue { double, double } %113, 1
  %118 = fmul double %114, %116
  %119 = fmul double %115, %117
  %120 = fmul double %115, %116
  %121 = fmul double %114, %117
  %122 = fsub double %118, %119
  %123 = fadd double %120, %121
  %124 = fadd double %107, %122
  %125 = fadd double %108, %123
  %126 = insertvalue { double, double } poison, double %124, 0
  %127 = insertvalue { double, double } %126, double %125, 1
  %128 = getelementptr inbounds [4096 x { double, double }], ptr %2, i32 0, i32 %9
  store { double, double } %127, ptr %128, align 8
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

attributes #0 = { "nvvm.reqntid"="128,1,1" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 0, i32 32}
!2 = !{i32 0, i32 128}
!3 = !{}
