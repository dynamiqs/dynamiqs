; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @loop_add_fusion_1(ptr noalias align 256 dereferenceable(65536) %0, ptr noalias align 256 dereferenceable(458752) %1, ptr noalias align 256 dereferenceable(784) %2, ptr noalias align 256 dereferenceable(8) %3, ptr noalias align 256 dereferenceable(8) %4, ptr noalias align 256 dereferenceable(65536) %5) #0 {
  %7 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !1
  %8 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %9 = mul i32 %7, 128
  %10 = add i32 %9, %8
  %11 = getelementptr inbounds [4096 x { double, double }], ptr %0, i32 0, i32 %10
  %12 = load { double, double }, ptr %11, align 8, !invariant.load !3
  %13 = getelementptr inbounds [1 x i64], ptr %3, i32 0, i32 0
  %14 = load i64, ptr %13, align 4, !invariant.load !3
  %15 = getelementptr inbounds [1 x i64], ptr %4, i32 0, i32 0
  %16 = load i64, ptr %15, align 4, !invariant.load !3
  %17 = add i64 %14, %16
  %18 = trunc i64 %17 to i32
  %19 = icmp slt i32 %18, 0
  %20 = add i32 %18, 7
  %21 = select i1 %19, i32 %20, i32 %18
  %22 = call i32 @llvm.smin.i32(i32 %21, i32 6)
  %23 = call i32 @llvm.smax.i32(i32 %22, i32 0)
  %24 = mul i32 %23, 7
  %25 = getelementptr inbounds [49 x { double, double }], ptr %2, i32 0, i32 %24
  %26 = load { double, double }, ptr %25, align 8, !invariant.load !3
  %27 = getelementptr inbounds [28672 x { double, double }], ptr %1, i32 0, i32 %10
  %28 = load { double, double }, ptr %27, align 8, !invariant.load !3
  %29 = extractvalue { double, double } %26, 0
  %30 = extractvalue { double, double } %26, 1
  %31 = extractvalue { double, double } %28, 0
  %32 = extractvalue { double, double } %28, 1
  %33 = fmul double %29, %31
  %34 = fmul double %30, %32
  %35 = fmul double %30, %31
  %36 = fmul double %29, %32
  %37 = fsub double %33, %34
  %38 = fadd double %35, %36
  %39 = fadd double %37, 0.000000e+00
  %40 = fadd double %38, 0.000000e+00
  %41 = add i32 %24, 1
  %42 = getelementptr inbounds [49 x { double, double }], ptr %2, i32 0, i32 %41
  %43 = load { double, double }, ptr %42, align 8, !invariant.load !3
  %44 = add i32 %10, 4096
  %45 = getelementptr inbounds [28672 x { double, double }], ptr %1, i32 0, i32 %44
  %46 = load { double, double }, ptr %45, align 8, !invariant.load !3
  %47 = extractvalue { double, double } %43, 0
  %48 = extractvalue { double, double } %43, 1
  %49 = extractvalue { double, double } %46, 0
  %50 = extractvalue { double, double } %46, 1
  %51 = fmul double %47, %49
  %52 = fmul double %48, %50
  %53 = fmul double %48, %49
  %54 = fmul double %47, %50
  %55 = fsub double %51, %52
  %56 = fadd double %53, %54
  %57 = fadd double %39, %55
  %58 = fadd double %40, %56
  %59 = add i32 %24, 2
  %60 = getelementptr inbounds [49 x { double, double }], ptr %2, i32 0, i32 %59
  %61 = load { double, double }, ptr %60, align 8, !invariant.load !3
  %62 = add i32 %10, 8192
  %63 = getelementptr inbounds [28672 x { double, double }], ptr %1, i32 0, i32 %62
  %64 = load { double, double }, ptr %63, align 8, !invariant.load !3
  %65 = extractvalue { double, double } %61, 0
  %66 = extractvalue { double, double } %61, 1
  %67 = extractvalue { double, double } %64, 0
  %68 = extractvalue { double, double } %64, 1
  %69 = fmul double %65, %67
  %70 = fmul double %66, %68
  %71 = fmul double %66, %67
  %72 = fmul double %65, %68
  %73 = fsub double %69, %70
  %74 = fadd double %71, %72
  %75 = fadd double %57, %73
  %76 = fadd double %58, %74
  %77 = add i32 %24, 3
  %78 = getelementptr inbounds [49 x { double, double }], ptr %2, i32 0, i32 %77
  %79 = load { double, double }, ptr %78, align 8, !invariant.load !3
  %80 = add i32 %10, 12288
  %81 = getelementptr inbounds [28672 x { double, double }], ptr %1, i32 0, i32 %80
  %82 = load { double, double }, ptr %81, align 8, !invariant.load !3
  %83 = extractvalue { double, double } %79, 0
  %84 = extractvalue { double, double } %79, 1
  %85 = extractvalue { double, double } %82, 0
  %86 = extractvalue { double, double } %82, 1
  %87 = fmul double %83, %85
  %88 = fmul double %84, %86
  %89 = fmul double %84, %85
  %90 = fmul double %83, %86
  %91 = fsub double %87, %88
  %92 = fadd double %89, %90
  %93 = fadd double %75, %91
  %94 = fadd double %76, %92
  %95 = add i32 %24, 4
  %96 = getelementptr inbounds [49 x { double, double }], ptr %2, i32 0, i32 %95
  %97 = load { double, double }, ptr %96, align 8, !invariant.load !3
  %98 = add i32 %10, 16384
  %99 = getelementptr inbounds [28672 x { double, double }], ptr %1, i32 0, i32 %98
  %100 = load { double, double }, ptr %99, align 8, !invariant.load !3
  %101 = extractvalue { double, double } %97, 0
  %102 = extractvalue { double, double } %97, 1
  %103 = extractvalue { double, double } %100, 0
  %104 = extractvalue { double, double } %100, 1
  %105 = fmul double %101, %103
  %106 = fmul double %102, %104
  %107 = fmul double %102, %103
  %108 = fmul double %101, %104
  %109 = fsub double %105, %106
  %110 = fadd double %107, %108
  %111 = fadd double %93, %109
  %112 = fadd double %94, %110
  %113 = add i32 %24, 5
  %114 = getelementptr inbounds [49 x { double, double }], ptr %2, i32 0, i32 %113
  %115 = load { double, double }, ptr %114, align 8, !invariant.load !3
  %116 = add i32 %10, 20480
  %117 = getelementptr inbounds [28672 x { double, double }], ptr %1, i32 0, i32 %116
  %118 = load { double, double }, ptr %117, align 8, !invariant.load !3
  %119 = extractvalue { double, double } %115, 0
  %120 = extractvalue { double, double } %115, 1
  %121 = extractvalue { double, double } %118, 0
  %122 = extractvalue { double, double } %118, 1
  %123 = fmul double %119, %121
  %124 = fmul double %120, %122
  %125 = fmul double %120, %121
  %126 = fmul double %119, %122
  %127 = fsub double %123, %124
  %128 = fadd double %125, %126
  %129 = fadd double %111, %127
  %130 = fadd double %112, %128
  %131 = add i32 %24, 6
  %132 = getelementptr inbounds [49 x { double, double }], ptr %2, i32 0, i32 %131
  %133 = load { double, double }, ptr %132, align 8, !invariant.load !3
  %134 = add i32 %10, 24576
  %135 = getelementptr inbounds [28672 x { double, double }], ptr %1, i32 0, i32 %134
  %136 = load { double, double }, ptr %135, align 8, !invariant.load !3
  %137 = extractvalue { double, double } %133, 0
  %138 = extractvalue { double, double } %133, 1
  %139 = extractvalue { double, double } %136, 0
  %140 = extractvalue { double, double } %136, 1
  %141 = fmul double %137, %139
  %142 = fmul double %138, %140
  %143 = fmul double %138, %139
  %144 = fmul double %137, %140
  %145 = fsub double %141, %142
  %146 = fadd double %143, %144
  %147 = fadd double %129, %145
  %148 = fadd double %130, %146
  %149 = extractvalue { double, double } %12, 0
  %150 = fadd double %149, %147
  %151 = extractvalue { double, double } %12, 1
  %152 = fadd double %151, %148
  %153 = insertvalue { double, double } poison, double %150, 0
  %154 = insertvalue { double, double } %153, double %152, 1
  %155 = getelementptr inbounds [4096 x { double, double }], ptr %5, i32 0, i32 %10
  store { double, double } %154, ptr %155, align 8
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
