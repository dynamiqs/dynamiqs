; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@shared_0 = private addrspace(3) global [1056 x double] undef

declare double @__nv_fabs(double)

declare double @__nv_sqrt(double)

define ptx_kernel void @input_transpose_fusion_2(ptr noalias align 256 dereferenceable(65536) %0, ptr noalias align 256 dereferenceable(32768) %1) #0 {
  %3 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !1
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !2
  %5 = urem i32 %4, 2
  %6 = mul i32 %5, 32
  %7 = udiv i32 %3, 32
  %8 = mul i32 %7, 64
  %9 = add i32 %6, %8
  %10 = udiv i32 %4, 2
  %11 = mul i32 %10, 2048
  %12 = add i32 %9, %11
  %13 = urem i32 %3, 32
  %14 = add i32 %12, %13
  %15 = getelementptr inbounds [4096 x { double, double }], ptr %0, i32 0, i32 %14
  %16 = load { double, double }, ptr %15, align 8, !invariant.load !3
  %17 = extractvalue { double, double } %16, 0
  %18 = extractvalue { double, double } %16, 1
  %19 = call double @__nv_fabs(double %17)
  %20 = call double @__nv_fabs(double %18)
  %21 = call double @llvm.maximum.f64(double %19, double %20)
  %22 = call double @llvm.minimum.f64(double %19, double %20)
  %23 = fdiv double %22, %21
  %24 = fmul double %23, %23
  %25 = fadd double %24, 1.000000e+00
  %26 = call double @__nv_sqrt(double %25)
  %27 = fmul double %21, %26
  %28 = fcmp uno double %27, %27
  %29 = select i1 %28, double %22, double %27
  %30 = mul i32 %13, 33
  %31 = add i32 %30, %7
  %32 = getelementptr inbounds [1056 x double], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %31
  store double %29, ptr %32, align 8
  %33 = add i32 %14, 256
  %34 = getelementptr inbounds [4096 x { double, double }], ptr %0, i32 0, i32 %33
  %35 = load { double, double }, ptr %34, align 8, !invariant.load !3
  %36 = extractvalue { double, double } %35, 0
  %37 = extractvalue { double, double } %35, 1
  %38 = call double @__nv_fabs(double %36)
  %39 = call double @__nv_fabs(double %37)
  %40 = call double @llvm.maximum.f64(double %38, double %39)
  %41 = call double @llvm.minimum.f64(double %38, double %39)
  %42 = fdiv double %41, %40
  %43 = fmul double %42, %42
  %44 = fadd double %43, 1.000000e+00
  %45 = call double @__nv_sqrt(double %44)
  %46 = fmul double %40, %45
  %47 = fcmp uno double %46, %46
  %48 = select i1 %47, double %41, double %46
  %49 = add i32 %31, 4
  %50 = getelementptr inbounds [1056 x double], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %49
  store double %48, ptr %50, align 8
  %51 = add i32 %14, 512
  %52 = getelementptr inbounds [4096 x { double, double }], ptr %0, i32 0, i32 %51
  %53 = load { double, double }, ptr %52, align 8, !invariant.load !3
  %54 = extractvalue { double, double } %53, 0
  %55 = extractvalue { double, double } %53, 1
  %56 = call double @__nv_fabs(double %54)
  %57 = call double @__nv_fabs(double %55)
  %58 = call double @llvm.maximum.f64(double %56, double %57)
  %59 = call double @llvm.minimum.f64(double %56, double %57)
  %60 = fdiv double %59, %58
  %61 = fmul double %60, %60
  %62 = fadd double %61, 1.000000e+00
  %63 = call double @__nv_sqrt(double %62)
  %64 = fmul double %58, %63
  %65 = fcmp uno double %64, %64
  %66 = select i1 %65, double %59, double %64
  %67 = add i32 %31, 8
  %68 = getelementptr inbounds [1056 x double], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %67
  store double %66, ptr %68, align 8
  %69 = add i32 %14, 768
  %70 = getelementptr inbounds [4096 x { double, double }], ptr %0, i32 0, i32 %69
  %71 = load { double, double }, ptr %70, align 8, !invariant.load !3
  %72 = extractvalue { double, double } %71, 0
  %73 = extractvalue { double, double } %71, 1
  %74 = call double @__nv_fabs(double %72)
  %75 = call double @__nv_fabs(double %73)
  %76 = call double @llvm.maximum.f64(double %74, double %75)
  %77 = call double @llvm.minimum.f64(double %74, double %75)
  %78 = fdiv double %77, %76
  %79 = fmul double %78, %78
  %80 = fadd double %79, 1.000000e+00
  %81 = call double @__nv_sqrt(double %80)
  %82 = fmul double %76, %81
  %83 = fcmp uno double %82, %82
  %84 = select i1 %83, double %77, double %82
  %85 = add i32 %31, 12
  %86 = getelementptr inbounds [1056 x double], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %85
  store double %84, ptr %86, align 8
  %87 = add i32 %14, 1024
  %88 = getelementptr inbounds [4096 x { double, double }], ptr %0, i32 0, i32 %87
  %89 = load { double, double }, ptr %88, align 8, !invariant.load !3
  %90 = extractvalue { double, double } %89, 0
  %91 = extractvalue { double, double } %89, 1
  %92 = call double @__nv_fabs(double %90)
  %93 = call double @__nv_fabs(double %91)
  %94 = call double @llvm.maximum.f64(double %92, double %93)
  %95 = call double @llvm.minimum.f64(double %92, double %93)
  %96 = fdiv double %95, %94
  %97 = fmul double %96, %96
  %98 = fadd double %97, 1.000000e+00
  %99 = call double @__nv_sqrt(double %98)
  %100 = fmul double %94, %99
  %101 = fcmp uno double %100, %100
  %102 = select i1 %101, double %95, double %100
  %103 = add i32 %31, 16
  %104 = getelementptr inbounds [1056 x double], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %103
  store double %102, ptr %104, align 8
  %105 = add i32 %14, 1280
  %106 = getelementptr inbounds [4096 x { double, double }], ptr %0, i32 0, i32 %105
  %107 = load { double, double }, ptr %106, align 8, !invariant.load !3
  %108 = extractvalue { double, double } %107, 0
  %109 = extractvalue { double, double } %107, 1
  %110 = call double @__nv_fabs(double %108)
  %111 = call double @__nv_fabs(double %109)
  %112 = call double @llvm.maximum.f64(double %110, double %111)
  %113 = call double @llvm.minimum.f64(double %110, double %111)
  %114 = fdiv double %113, %112
  %115 = fmul double %114, %114
  %116 = fadd double %115, 1.000000e+00
  %117 = call double @__nv_sqrt(double %116)
  %118 = fmul double %112, %117
  %119 = fcmp uno double %118, %118
  %120 = select i1 %119, double %113, double %118
  %121 = add i32 %31, 20
  %122 = getelementptr inbounds [1056 x double], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %121
  store double %120, ptr %122, align 8
  %123 = add i32 %14, 1536
  %124 = getelementptr inbounds [4096 x { double, double }], ptr %0, i32 0, i32 %123
  %125 = load { double, double }, ptr %124, align 8, !invariant.load !3
  %126 = extractvalue { double, double } %125, 0
  %127 = extractvalue { double, double } %125, 1
  %128 = call double @__nv_fabs(double %126)
  %129 = call double @__nv_fabs(double %127)
  %130 = call double @llvm.maximum.f64(double %128, double %129)
  %131 = call double @llvm.minimum.f64(double %128, double %129)
  %132 = fdiv double %131, %130
  %133 = fmul double %132, %132
  %134 = fadd double %133, 1.000000e+00
  %135 = call double @__nv_sqrt(double %134)
  %136 = fmul double %130, %135
  %137 = fcmp uno double %136, %136
  %138 = select i1 %137, double %131, double %136
  %139 = add i32 %31, 24
  %140 = getelementptr inbounds [1056 x double], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %139
  store double %138, ptr %140, align 8
  %141 = add i32 %14, 1792
  %142 = getelementptr inbounds [4096 x { double, double }], ptr %0, i32 0, i32 %141
  %143 = load { double, double }, ptr %142, align 8, !invariant.load !3
  %144 = extractvalue { double, double } %143, 0
  %145 = extractvalue { double, double } %143, 1
  %146 = call double @__nv_fabs(double %144)
  %147 = call double @__nv_fabs(double %145)
  %148 = call double @llvm.maximum.f64(double %146, double %147)
  %149 = call double @llvm.minimum.f64(double %146, double %147)
  %150 = fdiv double %149, %148
  %151 = fmul double %150, %150
  %152 = fadd double %151, 1.000000e+00
  %153 = call double @__nv_sqrt(double %152)
  %154 = fmul double %148, %153
  %155 = fcmp uno double %154, %154
  %156 = select i1 %155, double %149, double %154
  %157 = add i32 %31, 28
  %158 = getelementptr inbounds [1056 x double], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %157
  store double %156, ptr %158, align 8
  call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %159 = mul i32 %7, 33
  %160 = add i32 %159, %13
  %161 = getelementptr inbounds [1056 x double], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %160
  %162 = load double, ptr %161, align 8
  %163 = mul i32 %5, 2048
  %164 = add i32 %163, %8
  %165 = mul i32 %10, 32
  %166 = add i32 %164, %165
  %167 = add i32 %166, %13
  %168 = getelementptr inbounds [4096 x double], ptr %1, i32 0, i32 %167
  store double %162, ptr %168, align 8
  %169 = add i32 %160, 132
  %170 = getelementptr inbounds [1056 x double], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %169
  %171 = load double, ptr %170, align 8
  %172 = add i32 %167, 256
  %173 = getelementptr inbounds [4096 x double], ptr %1, i32 0, i32 %172
  store double %171, ptr %173, align 8
  %174 = add i32 %160, 264
  %175 = getelementptr inbounds [1056 x double], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %174
  %176 = load double, ptr %175, align 8
  %177 = add i32 %167, 512
  %178 = getelementptr inbounds [4096 x double], ptr %1, i32 0, i32 %177
  store double %176, ptr %178, align 8
  %179 = add i32 %160, 396
  %180 = getelementptr inbounds [1056 x double], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %179
  %181 = load double, ptr %180, align 8
  %182 = add i32 %167, 768
  %183 = getelementptr inbounds [4096 x double], ptr %1, i32 0, i32 %182
  store double %181, ptr %183, align 8
  %184 = add i32 %160, 528
  %185 = getelementptr inbounds [1056 x double], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %184
  %186 = load double, ptr %185, align 8
  %187 = add i32 %167, 1024
  %188 = getelementptr inbounds [4096 x double], ptr %1, i32 0, i32 %187
  store double %186, ptr %188, align 8
  %189 = add i32 %160, 660
  %190 = getelementptr inbounds [1056 x double], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %189
  %191 = load double, ptr %190, align 8
  %192 = add i32 %167, 1280
  %193 = getelementptr inbounds [4096 x double], ptr %1, i32 0, i32 %192
  store double %191, ptr %193, align 8
  %194 = add i32 %160, 792
  %195 = getelementptr inbounds [1056 x double], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %194
  %196 = load double, ptr %195, align 8
  %197 = add i32 %167, 1536
  %198 = getelementptr inbounds [4096 x double], ptr %1, i32 0, i32 %197
  store double %196, ptr %198, align 8
  %199 = add i32 %160, 924
  %200 = getelementptr inbounds [1056 x double], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %199
  %201 = load double, ptr %200, align 8
  %202 = add i32 %167, 1792
  %203 = getelementptr inbounds [4096 x double], ptr %1, i32 0, i32 %202
  store double %201, ptr %203, align 8
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.maximum.f64(double, double) #2

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.minimum.f64(double, double) #2

; Function Attrs: convergent nocallback nounwind
declare void @llvm.nvvm.barrier.cta.sync.aligned.all(i32) #3

attributes #0 = { "nvvm.reqntid"="128,1,1" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { convergent nocallback nounwind }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 0, i32 128}
!2 = !{i32 0, i32 4}
!3 = !{}
