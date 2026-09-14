; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @loop_reduce_fusion(ptr noalias align 256 dereferenceable(458752) %0, ptr noalias align 256 dereferenceable(784) %1, ptr noalias align 256 dereferenceable(8) %2, ptr noalias align 256 dereferenceable(8) %3, ptr noalias align 256 dereferenceable(65536) %4) #0 {
  %6 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !1
  %7 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %8 = getelementptr inbounds [1 x i64], ptr %2, i32 0, i32 0
  %9 = load i64, ptr %8, align 4, !invariant.load !3
  %10 = getelementptr inbounds [1 x i64], ptr %3, i32 0, i32 0
  %11 = load i64, ptr %10, align 4, !invariant.load !3
  %12 = add i64 %9, %11
  %13 = trunc i64 %12 to i32
  %14 = icmp slt i32 %13, 0
  %15 = add i32 %13, 7
  %16 = select i1 %14, i32 %15, i32 %13
  %17 = call i32 @llvm.smin.i32(i32 %16, i32 6)
  %18 = call i32 @llvm.smax.i32(i32 %17, i32 0)
  %19 = mul i32 %18, 7
  %20 = getelementptr inbounds [49 x { double, double }], ptr %1, i32 0, i32 %19
  %21 = load { double, double }, ptr %20, align 8, !invariant.load !3
  %22 = mul i32 %6, 128
  %23 = add i32 %22, %7
  %24 = getelementptr inbounds [28672 x { double, double }], ptr %0, i32 0, i32 %23
  %25 = load { double, double }, ptr %24, align 8, !invariant.load !3
  %26 = extractvalue { double, double } %21, 0
  %27 = extractvalue { double, double } %21, 1
  %28 = extractvalue { double, double } %25, 0
  %29 = extractvalue { double, double } %25, 1
  %30 = fmul double %26, %28
  %31 = fmul double %27, %29
  %32 = fmul double %27, %28
  %33 = fmul double %26, %29
  %34 = fsub double %30, %31
  %35 = fadd double %32, %33
  %36 = fadd double %34, 0.000000e+00
  %37 = fadd double %35, 0.000000e+00
  %38 = add i32 %19, 1
  %39 = getelementptr inbounds [49 x { double, double }], ptr %1, i32 0, i32 %38
  %40 = load { double, double }, ptr %39, align 8, !invariant.load !3
  %41 = add i32 %23, 4096
  %42 = getelementptr inbounds [28672 x { double, double }], ptr %0, i32 0, i32 %41
  %43 = load { double, double }, ptr %42, align 8, !invariant.load !3
  %44 = extractvalue { double, double } %40, 0
  %45 = extractvalue { double, double } %40, 1
  %46 = extractvalue { double, double } %43, 0
  %47 = extractvalue { double, double } %43, 1
  %48 = fmul double %44, %46
  %49 = fmul double %45, %47
  %50 = fmul double %45, %46
  %51 = fmul double %44, %47
  %52 = fsub double %48, %49
  %53 = fadd double %50, %51
  %54 = fadd double %36, %52
  %55 = fadd double %37, %53
  %56 = add i32 %19, 2
  %57 = getelementptr inbounds [49 x { double, double }], ptr %1, i32 0, i32 %56
  %58 = load { double, double }, ptr %57, align 8, !invariant.load !3
  %59 = add i32 %23, 8192
  %60 = getelementptr inbounds [28672 x { double, double }], ptr %0, i32 0, i32 %59
  %61 = load { double, double }, ptr %60, align 8, !invariant.load !3
  %62 = extractvalue { double, double } %58, 0
  %63 = extractvalue { double, double } %58, 1
  %64 = extractvalue { double, double } %61, 0
  %65 = extractvalue { double, double } %61, 1
  %66 = fmul double %62, %64
  %67 = fmul double %63, %65
  %68 = fmul double %63, %64
  %69 = fmul double %62, %65
  %70 = fsub double %66, %67
  %71 = fadd double %68, %69
  %72 = fadd double %54, %70
  %73 = fadd double %55, %71
  %74 = add i32 %19, 3
  %75 = getelementptr inbounds [49 x { double, double }], ptr %1, i32 0, i32 %74
  %76 = load { double, double }, ptr %75, align 8, !invariant.load !3
  %77 = add i32 %23, 12288
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
  %90 = fadd double %72, %88
  %91 = fadd double %73, %89
  %92 = add i32 %19, 4
  %93 = getelementptr inbounds [49 x { double, double }], ptr %1, i32 0, i32 %92
  %94 = load { double, double }, ptr %93, align 8, !invariant.load !3
  %95 = add i32 %23, 16384
  %96 = getelementptr inbounds [28672 x { double, double }], ptr %0, i32 0, i32 %95
  %97 = load { double, double }, ptr %96, align 8, !invariant.load !3
  %98 = extractvalue { double, double } %94, 0
  %99 = extractvalue { double, double } %94, 1
  %100 = extractvalue { double, double } %97, 0
  %101 = extractvalue { double, double } %97, 1
  %102 = fmul double %98, %100
  %103 = fmul double %99, %101
  %104 = fmul double %99, %100
  %105 = fmul double %98, %101
  %106 = fsub double %102, %103
  %107 = fadd double %104, %105
  %108 = fadd double %90, %106
  %109 = fadd double %91, %107
  %110 = add i32 %19, 5
  %111 = getelementptr inbounds [49 x { double, double }], ptr %1, i32 0, i32 %110
  %112 = load { double, double }, ptr %111, align 8, !invariant.load !3
  %113 = add i32 %23, 20480
  %114 = getelementptr inbounds [28672 x { double, double }], ptr %0, i32 0, i32 %113
  %115 = load { double, double }, ptr %114, align 8, !invariant.load !3
  %116 = extractvalue { double, double } %112, 0
  %117 = extractvalue { double, double } %112, 1
  %118 = extractvalue { double, double } %115, 0
  %119 = extractvalue { double, double } %115, 1
  %120 = fmul double %116, %118
  %121 = fmul double %117, %119
  %122 = fmul double %117, %118
  %123 = fmul double %116, %119
  %124 = fsub double %120, %121
  %125 = fadd double %122, %123
  %126 = fadd double %108, %124
  %127 = fadd double %109, %125
  %128 = add i32 %19, 6
  %129 = getelementptr inbounds [49 x { double, double }], ptr %1, i32 0, i32 %128
  %130 = load { double, double }, ptr %129, align 8, !invariant.load !3
  %131 = add i32 %23, 24576
  %132 = getelementptr inbounds [28672 x { double, double }], ptr %0, i32 0, i32 %131
  %133 = load { double, double }, ptr %132, align 8, !invariant.load !3
  %134 = extractvalue { double, double } %130, 0
  %135 = extractvalue { double, double } %130, 1
  %136 = extractvalue { double, double } %133, 0
  %137 = extractvalue { double, double } %133, 1
  %138 = fmul double %134, %136
  %139 = fmul double %135, %137
  %140 = fmul double %135, %136
  %141 = fmul double %134, %137
  %142 = fsub double %138, %139
  %143 = fadd double %140, %141
  %144 = fadd double %126, %142
  %145 = fadd double %127, %143
  %146 = insertvalue { double, double } poison, double %144, 0
  %147 = insertvalue { double, double } %146, double %145, 1
  %148 = getelementptr inbounds [4096 x { double, double }], ptr %4, i32 0, i32 %23
  store { double, double } %147, ptr %148, align 8
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.smin.i32(i32, i32) #2

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.smax.i32(i32, i32) #2

attributes #0 = { "nvvm.reqntid"="128,1,1" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 0, i32 32}
!2 = !{i32 0, i32 128}
!3 = !{}
