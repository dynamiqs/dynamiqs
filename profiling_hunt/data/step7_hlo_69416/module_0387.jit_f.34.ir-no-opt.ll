; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @loop_dynamic_update_slice_fusion(ptr noalias align 256 dereferenceable(10485760) %0, ptr noalias align 256 dereferenceable(65536) %1, ptr noalias align 256 dereferenceable(458752) %2, ptr noalias align 256 dereferenceable(56) %3, ptr noalias align 256 dereferenceable(8) %4, ptr noalias align 256 dereferenceable(8) %5, ptr noalias align 256 dereferenceable(12) %6, ptr noalias align 256 dereferenceable(10485760) %7) #0 {
  %9 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !1
  %10 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %11 = getelementptr inbounds [1 x i64], ptr %4, i32 0, i32 0
  %12 = load i64, ptr %11, align 4, !invariant.load !3
  %13 = getelementptr inbounds [1 x i64], ptr %5, i32 0, i32 0
  %14 = load i64, ptr %13, align 4, !invariant.load !3
  %15 = add i64 %12, %14
  %16 = icmp slt i64 %15, 0
  %17 = add i64 %15, 160
  %18 = select i1 %16, i64 %17, i64 %15
  %19 = trunc i64 %18 to i32
  %20 = call i32 @llvm.smin.i32(i32 %19, i32 159)
  %21 = call i32 @llvm.smax.i32(i32 %20, i32 0)
  %22 = getelementptr inbounds [3 x i32], ptr %6, i32 0, i32 0
  %23 = load i32, ptr %22, align 4, !invariant.load !3
  %24 = icmp sge i32 %19, 0
  %25 = icmp sge i32 %23, %19
  %26 = and i1 %24, %25
  %27 = zext i1 %26 to i8
  %28 = and i8 %27, 1
  %29 = getelementptr inbounds [3 x i32], ptr %6, i32 0, i32 1
  %30 = load i32, ptr %29, align 4, !invariant.load !3
  %31 = icmp sge i32 %30, 0
  %32 = zext i1 %31 to i8
  %33 = and i8 %28, %32
  %34 = getelementptr inbounds [3 x i32], ptr %6, i32 0, i32 2
  %35 = load i32, ptr %34, align 4, !invariant.load !3
  %36 = icmp sge i32 %35, 0
  %37 = zext i1 %36 to i8
  %38 = and i8 %33, %37
  %39 = mul i32 %9, 128
  %40 = add i32 %39, %10
  %41 = getelementptr inbounds [4096 x { double, double }], ptr %1, i32 0, i32 %40
  %42 = load { double, double }, ptr %41, align 8, !invariant.load !3
  %43 = getelementptr inbounds [7 x double], ptr %3, i32 0, i32 0
  %44 = load double, ptr %43, align 8, !invariant.load !3
  %45 = getelementptr inbounds [28672 x { double, double }], ptr %2, i32 0, i32 %40
  %46 = load { double, double }, ptr %45, align 8, !invariant.load !3
  %47 = extractvalue { double, double } %46, 0
  %48 = extractvalue { double, double } %46, 1
  %49 = fmul double %44, %47
  %50 = fmul double %48, 0.000000e+00
  %51 = fmul double %47, 0.000000e+00
  %52 = fmul double %44, %48
  %53 = fsub double %49, %50
  %54 = fadd double %51, %52
  %55 = fadd double %53, 0.000000e+00
  %56 = fadd double %54, 0.000000e+00
  %57 = getelementptr inbounds [7 x double], ptr %3, i32 0, i32 1
  %58 = load double, ptr %57, align 8, !invariant.load !3
  %59 = add i32 %40, 4096
  %60 = getelementptr inbounds [28672 x { double, double }], ptr %2, i32 0, i32 %59
  %61 = load { double, double }, ptr %60, align 8, !invariant.load !3
  %62 = extractvalue { double, double } %61, 0
  %63 = extractvalue { double, double } %61, 1
  %64 = fmul double %58, %62
  %65 = fmul double %63, 0.000000e+00
  %66 = fmul double %62, 0.000000e+00
  %67 = fmul double %58, %63
  %68 = fsub double %64, %65
  %69 = fadd double %66, %67
  %70 = fadd double %55, %68
  %71 = fadd double %56, %69
  %72 = getelementptr inbounds [7 x double], ptr %3, i32 0, i32 2
  %73 = load double, ptr %72, align 8, !invariant.load !3
  %74 = add i32 %40, 8192
  %75 = getelementptr inbounds [28672 x { double, double }], ptr %2, i32 0, i32 %74
  %76 = load { double, double }, ptr %75, align 8, !invariant.load !3
  %77 = extractvalue { double, double } %76, 0
  %78 = extractvalue { double, double } %76, 1
  %79 = fmul double %73, %77
  %80 = fmul double %78, 0.000000e+00
  %81 = fmul double %77, 0.000000e+00
  %82 = fmul double %73, %78
  %83 = fsub double %79, %80
  %84 = fadd double %81, %82
  %85 = fadd double %70, %83
  %86 = fadd double %71, %84
  %87 = getelementptr inbounds [7 x double], ptr %3, i32 0, i32 3
  %88 = load double, ptr %87, align 8, !invariant.load !3
  %89 = add i32 %40, 12288
  %90 = getelementptr inbounds [28672 x { double, double }], ptr %2, i32 0, i32 %89
  %91 = load { double, double }, ptr %90, align 8, !invariant.load !3
  %92 = extractvalue { double, double } %91, 0
  %93 = extractvalue { double, double } %91, 1
  %94 = fmul double %88, %92
  %95 = fmul double %93, 0.000000e+00
  %96 = fmul double %92, 0.000000e+00
  %97 = fmul double %88, %93
  %98 = fsub double %94, %95
  %99 = fadd double %96, %97
  %100 = fadd double %85, %98
  %101 = fadd double %86, %99
  %102 = getelementptr inbounds [7 x double], ptr %3, i32 0, i32 4
  %103 = load double, ptr %102, align 8, !invariant.load !3
  %104 = add i32 %40, 16384
  %105 = getelementptr inbounds [28672 x { double, double }], ptr %2, i32 0, i32 %104
  %106 = load { double, double }, ptr %105, align 8, !invariant.load !3
  %107 = extractvalue { double, double } %106, 0
  %108 = extractvalue { double, double } %106, 1
  %109 = fmul double %103, %107
  %110 = fmul double %108, 0.000000e+00
  %111 = fmul double %107, 0.000000e+00
  %112 = fmul double %103, %108
  %113 = fsub double %109, %110
  %114 = fadd double %111, %112
  %115 = fadd double %100, %113
  %116 = fadd double %101, %114
  %117 = getelementptr inbounds [7 x double], ptr %3, i32 0, i32 5
  %118 = load double, ptr %117, align 8, !invariant.load !3
  %119 = add i32 %40, 20480
  %120 = getelementptr inbounds [28672 x { double, double }], ptr %2, i32 0, i32 %119
  %121 = load { double, double }, ptr %120, align 8, !invariant.load !3
  %122 = extractvalue { double, double } %121, 0
  %123 = extractvalue { double, double } %121, 1
  %124 = fmul double %118, %122
  %125 = fmul double %123, 0.000000e+00
  %126 = fmul double %122, 0.000000e+00
  %127 = fmul double %118, %123
  %128 = fsub double %124, %125
  %129 = fadd double %126, %127
  %130 = fadd double %115, %128
  %131 = fadd double %116, %129
  %132 = getelementptr inbounds [7 x double], ptr %3, i32 0, i32 6
  %133 = load double, ptr %132, align 8, !invariant.load !3
  %134 = add i32 %40, 24576
  %135 = getelementptr inbounds [28672 x { double, double }], ptr %2, i32 0, i32 %134
  %136 = load { double, double }, ptr %135, align 8, !invariant.load !3
  %137 = extractvalue { double, double } %136, 0
  %138 = extractvalue { double, double } %136, 1
  %139 = fmul double %133, %137
  %140 = fmul double %138, 0.000000e+00
  %141 = fmul double %137, 0.000000e+00
  %142 = fmul double %133, %138
  %143 = fsub double %139, %140
  %144 = fadd double %141, %142
  %145 = fadd double %130, %143
  %146 = fadd double %131, %144
  %147 = extractvalue { double, double } %42, 0
  %148 = fadd double %147, %145
  %149 = extractvalue { double, double } %42, 1
  %150 = fadd double %149, %146
  %151 = insertvalue { double, double } poison, double %148, 0
  %152 = insertvalue { double, double } %151, double %150, 1
  %153 = mul i32 %21, 4096
  %154 = add i32 %153, %39
  %155 = add i32 %154, %10
  %156 = getelementptr inbounds [655360 x { double, double }], ptr %0, i32 0, i32 %155
  %157 = load { double, double }, ptr %156, align 8
  %158 = trunc i8 %38 to i1
  %159 = select i1 %158, { double, double } %152, { double, double } %157
  store { double, double } %159, ptr %156, align 8
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
