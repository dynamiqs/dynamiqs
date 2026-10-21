; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@shared_0 = private addrspace(3) global [8 x i8] undef

define ptx_kernel void @input_reduce_fusion_1(ptr noalias align 256 dereferenceable(65536) %0, ptr noalias align 256 dereferenceable(1) %1) #0 {
  %3 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !1
  %4 = getelementptr inbounds [4096 x { double, double }], ptr %0, i32 0, i32 %3
  %5 = load { double, double }, ptr %4, align 8, !invariant.load !2
  %6 = extractvalue { double, double } %5, 0
  %7 = extractvalue { double, double } %5, 1
  %8 = fcmp une double %6, %6
  %9 = fcmp une double %7, %7
  %10 = or i1 %8, %9
  %11 = zext i1 %10 to i8
  %12 = call i8 @region_3_8_clone_1_reduce_or_32_0(i8 0, i8 %11)
  %13 = add i32 %3, 256
  %14 = getelementptr inbounds [4096 x { double, double }], ptr %0, i32 0, i32 %13
  %15 = load { double, double }, ptr %14, align 8, !invariant.load !2
  %16 = extractvalue { double, double } %15, 0
  %17 = extractvalue { double, double } %15, 1
  %18 = fcmp une double %16, %16
  %19 = fcmp une double %17, %17
  %20 = or i1 %18, %19
  %21 = zext i1 %20 to i8
  %22 = call i8 @region_3_8_clone_1_reduce_or_32_0(i8 %12, i8 %21)
  %23 = add i32 %3, 512
  %24 = getelementptr inbounds [4096 x { double, double }], ptr %0, i32 0, i32 %23
  %25 = load { double, double }, ptr %24, align 8, !invariant.load !2
  %26 = extractvalue { double, double } %25, 0
  %27 = extractvalue { double, double } %25, 1
  %28 = fcmp une double %26, %26
  %29 = fcmp une double %27, %27
  %30 = or i1 %28, %29
  %31 = zext i1 %30 to i8
  %32 = call i8 @region_3_8_clone_1_reduce_or_32_0(i8 %22, i8 %31)
  %33 = add i32 %3, 768
  %34 = getelementptr inbounds [4096 x { double, double }], ptr %0, i32 0, i32 %33
  %35 = load { double, double }, ptr %34, align 8, !invariant.load !2
  %36 = extractvalue { double, double } %35, 0
  %37 = extractvalue { double, double } %35, 1
  %38 = fcmp une double %36, %36
  %39 = fcmp une double %37, %37
  %40 = or i1 %38, %39
  %41 = zext i1 %40 to i8
  %42 = call i8 @region_3_8_clone_1_reduce_or_32_0(i8 %32, i8 %41)
  %43 = add i32 %3, 1024
  %44 = getelementptr inbounds [4096 x { double, double }], ptr %0, i32 0, i32 %43
  %45 = load { double, double }, ptr %44, align 8, !invariant.load !2
  %46 = extractvalue { double, double } %45, 0
  %47 = extractvalue { double, double } %45, 1
  %48 = fcmp une double %46, %46
  %49 = fcmp une double %47, %47
  %50 = or i1 %48, %49
  %51 = zext i1 %50 to i8
  %52 = call i8 @region_3_8_clone_1_reduce_or_32_0(i8 %42, i8 %51)
  %53 = add i32 %3, 1280
  %54 = getelementptr inbounds [4096 x { double, double }], ptr %0, i32 0, i32 %53
  %55 = load { double, double }, ptr %54, align 8, !invariant.load !2
  %56 = extractvalue { double, double } %55, 0
  %57 = extractvalue { double, double } %55, 1
  %58 = fcmp une double %56, %56
  %59 = fcmp une double %57, %57
  %60 = or i1 %58, %59
  %61 = zext i1 %60 to i8
  %62 = call i8 @region_3_8_clone_1_reduce_or_32_0(i8 %52, i8 %61)
  %63 = add i32 %3, 1536
  %64 = getelementptr inbounds [4096 x { double, double }], ptr %0, i32 0, i32 %63
  %65 = load { double, double }, ptr %64, align 8, !invariant.load !2
  %66 = extractvalue { double, double } %65, 0
  %67 = extractvalue { double, double } %65, 1
  %68 = fcmp une double %66, %66
  %69 = fcmp une double %67, %67
  %70 = or i1 %68, %69
  %71 = zext i1 %70 to i8
  %72 = call i8 @region_3_8_clone_1_reduce_or_32_0(i8 %62, i8 %71)
  %73 = add i32 %3, 1792
  %74 = getelementptr inbounds [4096 x { double, double }], ptr %0, i32 0, i32 %73
  %75 = load { double, double }, ptr %74, align 8, !invariant.load !2
  %76 = extractvalue { double, double } %75, 0
  %77 = extractvalue { double, double } %75, 1
  %78 = fcmp une double %76, %76
  %79 = fcmp une double %77, %77
  %80 = or i1 %78, %79
  %81 = zext i1 %80 to i8
  %82 = call i8 @region_3_8_clone_1_reduce_or_32_0(i8 %72, i8 %81)
  %83 = add i32 %3, 2048
  %84 = getelementptr inbounds [4096 x { double, double }], ptr %0, i32 0, i32 %83
  %85 = load { double, double }, ptr %84, align 8, !invariant.load !2
  %86 = extractvalue { double, double } %85, 0
  %87 = extractvalue { double, double } %85, 1
  %88 = fcmp une double %86, %86
  %89 = fcmp une double %87, %87
  %90 = or i1 %88, %89
  %91 = zext i1 %90 to i8
  %92 = call i8 @region_3_8_clone_1_reduce_or_32_0(i8 %82, i8 %91)
  %93 = add i32 %3, 2304
  %94 = getelementptr inbounds [4096 x { double, double }], ptr %0, i32 0, i32 %93
  %95 = load { double, double }, ptr %94, align 8, !invariant.load !2
  %96 = extractvalue { double, double } %95, 0
  %97 = extractvalue { double, double } %95, 1
  %98 = fcmp une double %96, %96
  %99 = fcmp une double %97, %97
  %100 = or i1 %98, %99
  %101 = zext i1 %100 to i8
  %102 = call i8 @region_3_8_clone_1_reduce_or_32_0(i8 %92, i8 %101)
  %103 = add i32 %3, 2560
  %104 = getelementptr inbounds [4096 x { double, double }], ptr %0, i32 0, i32 %103
  %105 = load { double, double }, ptr %104, align 8, !invariant.load !2
  %106 = extractvalue { double, double } %105, 0
  %107 = extractvalue { double, double } %105, 1
  %108 = fcmp une double %106, %106
  %109 = fcmp une double %107, %107
  %110 = or i1 %108, %109
  %111 = zext i1 %110 to i8
  %112 = call i8 @region_3_8_clone_1_reduce_or_32_0(i8 %102, i8 %111)
  %113 = add i32 %3, 2816
  %114 = getelementptr inbounds [4096 x { double, double }], ptr %0, i32 0, i32 %113
  %115 = load { double, double }, ptr %114, align 8, !invariant.load !2
  %116 = extractvalue { double, double } %115, 0
  %117 = extractvalue { double, double } %115, 1
  %118 = fcmp une double %116, %116
  %119 = fcmp une double %117, %117
  %120 = or i1 %118, %119
  %121 = zext i1 %120 to i8
  %122 = call i8 @region_3_8_clone_1_reduce_or_32_0(i8 %112, i8 %121)
  %123 = add i32 %3, 3072
  %124 = getelementptr inbounds [4096 x { double, double }], ptr %0, i32 0, i32 %123
  %125 = load { double, double }, ptr %124, align 8, !invariant.load !2
  %126 = extractvalue { double, double } %125, 0
  %127 = extractvalue { double, double } %125, 1
  %128 = fcmp une double %126, %126
  %129 = fcmp une double %127, %127
  %130 = or i1 %128, %129
  %131 = zext i1 %130 to i8
  %132 = call i8 @region_3_8_clone_1_reduce_or_32_0(i8 %122, i8 %131)
  %133 = add i32 %3, 3328
  %134 = getelementptr inbounds [4096 x { double, double }], ptr %0, i32 0, i32 %133
  %135 = load { double, double }, ptr %134, align 8, !invariant.load !2
  %136 = extractvalue { double, double } %135, 0
  %137 = extractvalue { double, double } %135, 1
  %138 = fcmp une double %136, %136
  %139 = fcmp une double %137, %137
  %140 = or i1 %138, %139
  %141 = zext i1 %140 to i8
  %142 = call i8 @region_3_8_clone_1_reduce_or_32_0(i8 %132, i8 %141)
  %143 = add i32 %3, 3584
  %144 = getelementptr inbounds [4096 x { double, double }], ptr %0, i32 0, i32 %143
  %145 = load { double, double }, ptr %144, align 8, !invariant.load !2
  %146 = extractvalue { double, double } %145, 0
  %147 = extractvalue { double, double } %145, 1
  %148 = fcmp une double %146, %146
  %149 = fcmp une double %147, %147
  %150 = or i1 %148, %149
  %151 = zext i1 %150 to i8
  %152 = call i8 @region_3_8_clone_1_reduce_or_32_0(i8 %142, i8 %151)
  %153 = add i32 %3, 3840
  %154 = getelementptr inbounds [4096 x { double, double }], ptr %0, i32 0, i32 %153
  %155 = load { double, double }, ptr %154, align 8, !invariant.load !2
  %156 = extractvalue { double, double } %155, 0
  %157 = extractvalue { double, double } %155, 1
  %158 = fcmp une double %156, %156
  %159 = fcmp une double %157, %157
  %160 = or i1 %158, %159
  %161 = zext i1 %160 to i8
  %162 = call i8 @region_3_8_clone_1_reduce_or_32_0(i8 %152, i8 %161)
  %163 = zext i8 %162 to i32
  %164 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %163, i32 16, i32 31)
  %165 = trunc i32 %164 to i8
  %166 = call i8 @region_3_8_clone_1_reduce_or_32_0(i8 %162, i8 %165)
  %167 = zext i8 %166 to i32
  %168 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %167, i32 8, i32 31)
  %169 = trunc i32 %168 to i8
  %170 = call i8 @region_3_8_clone_1_reduce_or_32_0(i8 %166, i8 %169)
  %171 = zext i8 %170 to i32
  %172 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %171, i32 4, i32 31)
  %173 = trunc i32 %172 to i8
  %174 = call i8 @region_3_8_clone_1_reduce_or_32_0(i8 %170, i8 %173)
  %175 = zext i8 %174 to i32
  %176 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %175, i32 2, i32 31)
  %177 = trunc i32 %176 to i8
  %178 = call i8 @region_3_8_clone_1_reduce_or_32_0(i8 %174, i8 %177)
  %179 = zext i8 %178 to i32
  %180 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %179, i32 1, i32 31)
  %181 = trunc i32 %180 to i8
  %182 = call i8 @region_3_8_clone_1_reduce_or_32_0(i8 %178, i8 %181)
  %183 = urem i32 %3, 32
  %184 = icmp eq i32 %183, 0
  br i1 %184, label %185, label %188

185:                                              ; preds = %2
  %186 = udiv i32 %3, 32
  %187 = getelementptr inbounds [8 x i8], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %186
  store i8 %182, ptr %187, align 1
  br label %188

188:                                              ; preds = %185, %2
  call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %189 = icmp sle i32 %3, 31
  br i1 %189, label %190, label %223

190:                                              ; preds = %188
  %191 = icmp sle i32 %3, 7
  br i1 %191, label %192, label %195

192:                                              ; preds = %190
  %193 = getelementptr inbounds [8 x i8], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %3
  %194 = load i8, ptr %193, align 1
  br label %196

195:                                              ; preds = %190
  br label %196

196:                                              ; preds = %192, %195
  %197 = phi i8 [ 0, %195 ], [ %194, %192 ]
  br label %198

198:                                              ; preds = %196
  %199 = zext i8 %197 to i32
  %200 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %199, i32 16, i32 31)
  %201 = trunc i32 %200 to i8
  %202 = call i8 @region_3_8_clone_1_reduce_or_32_0(i8 %197, i8 %201)
  %203 = zext i8 %202 to i32
  %204 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %203, i32 8, i32 31)
  %205 = trunc i32 %204 to i8
  %206 = call i8 @region_3_8_clone_1_reduce_or_32_0(i8 %202, i8 %205)
  %207 = zext i8 %206 to i32
  %208 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %207, i32 4, i32 31)
  %209 = trunc i32 %208 to i8
  %210 = call i8 @region_3_8_clone_1_reduce_or_32_0(i8 %206, i8 %209)
  %211 = zext i8 %210 to i32
  %212 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %211, i32 2, i32 31)
  %213 = trunc i32 %212 to i8
  %214 = call i8 @region_3_8_clone_1_reduce_or_32_0(i8 %210, i8 %213)
  %215 = zext i8 %214 to i32
  %216 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %215, i32 1, i32 31)
  %217 = trunc i32 %216 to i8
  %218 = call i8 @region_3_8_clone_1_reduce_or_32_0(i8 %214, i8 %217)
  %219 = icmp eq i32 %3, 0
  br i1 %219, label %220, label %222

220:                                              ; preds = %198
  %221 = getelementptr inbounds [1 x i8], ptr %1, i32 0, i32 0
  store i8 %218, ptr %221, align 1
  br label %222

222:                                              ; preds = %220, %198
  br label %223

223:                                              ; preds = %222, %188
  ret void
}

define internal i8 @region_3_8_clone_1_reduce_or_32_0(i8 %0, i8 %1) {
  %3 = or i8 %0, %1
  ret i8 %3
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

; Function Attrs: convergent nocallback nounwind memory(inaccessiblemem: readwrite)
declare i32 @llvm.nvvm.shfl.sync.down.i32(i32, i32, i32, i32) #2

; Function Attrs: convergent nocallback nounwind
declare void @llvm.nvvm.barrier.cta.sync.aligned.all(i32) #3

attributes #0 = { "nvvm.reqntid"="256,1,1" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { convergent nocallback nounwind memory(inaccessiblemem: readwrite) }
attributes #3 = { convergent nocallback nounwind }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 0, i32 256}
!2 = !{}
