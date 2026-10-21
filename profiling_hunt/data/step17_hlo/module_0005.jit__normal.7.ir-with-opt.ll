; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @loop_multiply_fusion(ptr noalias nofree readonly align 256 captures(none) dereferenceable(32768) %0, ptr noalias nofree readonly align 256 captures(none) dereferenceable(32768) %1, ptr noalias nofree writeonly align 256 captures(none) dereferenceable(65536) %2) local_unnamed_addr #0 {
  %4 = addrspacecast ptr %1 to ptr addrspace(1)
  %5 = addrspacecast ptr %0 to ptr addrspace(1)
  %6 = addrspacecast ptr %2 to ptr addrspace(1)
  %7 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !4
  %8 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !5
  %9 = shl nuw nsw i32 %7, 7
  %10 = or disjoint i32 %9, %8
  %11 = zext nneg i32 %10 to i64
  %12 = getelementptr inbounds [8 x i8], ptr addrspace(1) %4, i64 %11
  %13 = load double, ptr addrspace(1) %12, align 8, !invariant.load !6
  %14 = fneg double %13
  %15 = fmul double %13, %14
  %16 = tail call i32 @llvm.nvvm.d2i.hi(double %15) #5
  %17 = bitcast i32 %16 to float
  %18 = fcmp olt float %17, f0x3FE55555
  %19 = fcmp ogt float %17, f0xBFD99999
  %or.cond.i = and i1 %18, %19
  br i1 %or.cond.i, label %20, label %37

20:                                               ; preds = %3
  %21 = fadd double %15, 2.000000e+00
  %22 = fdiv double %15, %21
  %23 = fneg double %15
  %24 = fmul double %22, %23
  %25 = fadd double %15, %24
  %26 = fmul double %25, %25
  %27 = tail call double @llvm.fma.f64(double %26, double f0x3EB372FB2FBE14B5, double f0x3ED087FFCEB2DC44)
  %28 = tail call double @llvm.fma.f64(double %27, double %26, double f0x3EF3B9FF890F468C)
  %29 = tail call double @llvm.fma.f64(double %28, double %26, double f0x3F17457EFD51BAF8)
  %30 = tail call double @llvm.fma.f64(double %29, double %26, double f0x3F3C71C8DE3CE825)
  %31 = tail call double @llvm.fma.f64(double %30, double %26, double f0x3F6249248FA4661F)
  %32 = tail call double @llvm.fma.f64(double %31, double %26, double f0x3F899999999D70C4)
  %33 = tail call double @llvm.fma.f64(double %32, double %26, double f0x3FB5555555555462)
  %34 = fmul double %26, %33
  %35 = tail call double @llvm.fma.f64(double %34, double %25, double %24)
  %36 = fadd double %15, %35
  br label %__nv_log1p.exit

37:                                               ; preds = %3
  %38 = fadd double %15, 1.000000e+00
  %39 = tail call i32 @llvm.nvvm.d2i.hi(double %38) #5
  %40 = tail call i32 @llvm.nvvm.d2i.lo(double %38) #5
  %41 = icmp slt i32 %39, 1048576
  %42 = fmul double %38, f0x4350000000000000
  %43 = tail call i32 @llvm.nvvm.d2i.lo(double %42) #5
  %44 = tail call i32 @llvm.nvvm.d2i.hi(double %42) #5
  br i1 %41, label %45, label %46

45:                                               ; preds = %37
  br label %46

46:                                               ; preds = %45, %37
  %.0.i = phi double [ %42, %45 ], [ %38, %37 ]
  %ihi.i.0.i = phi i32 [ %44, %45 ], [ %39, %37 ]
  %ilo.i.0.i = phi i32 [ %43, %45 ], [ %40, %37 ]
  %e.i.0.i = phi i32 [ -1077, %45 ], [ -1023, %37 ]
  %47 = add i32 %ihi.i.0.i, -1
  %or.cond2.i = icmp ult i32 %47, 2146435071
  br i1 %or.cond2.i, label %48, label %95

48:                                               ; preds = %46
  %49 = lshr i32 %ihi.i.0.i, 20
  %50 = add nsw i32 %e.i.0.i, %49
  %51 = and i32 %ihi.i.0.i, 1048575
  %52 = or disjoint i32 %51, 1072693248
  %53 = tail call double @llvm.nvvm.lohi.i2d(i32 %ilo.i.0.i, i32 %52) #5
  %54 = icmp samesign ugt i32 %52, 1073127582
  %55 = tail call i32 @llvm.nvvm.d2i.lo(double %53) #5
  %56 = tail call i32 @llvm.nvvm.d2i.hi(double %53) #5
  %57 = add i32 %56, -1048576
  %58 = tail call double @llvm.nvvm.lohi.i2d(i32 %55, i32 %57) #5
  %spec.select = select i1 %54, double %58, double %53
  %59 = zext i1 %54 to i32
  %spec.select3 = add nsw i32 %50, %59
  %60 = fadd double %spec.select, -1.000000e+00
  %61 = fadd double %spec.select, 1.000000e+00
  %62 = tail call double @llvm.nvvm.rcp.approx.ftz.d(double %61) #5
  %63 = fneg double %61
  %64 = tail call double @llvm.fma.f64(double %63, double %62, double 1.000000e+00)
  %65 = tail call double @llvm.fma.f64(double %64, double %64, double %64)
  %66 = tail call double @llvm.fma.f64(double %65, double %62, double %62)
  %67 = fmul double %60, %66
  %68 = fadd double %67, %67
  %69 = fmul double %68, %68
  %70 = tail call double @llvm.fma.f64(double %69, double f0x3EB1380B3AE80F1E, double f0x3ED0EE258B7A8B04)
  %71 = tail call double @llvm.fma.f64(double %70, double %69, double f0x3EF3B2669F02676F)
  %72 = tail call double @llvm.fma.f64(double %71, double %69, double f0x3F1745CBA9AB0956)
  %73 = tail call double @llvm.fma.f64(double %72, double %69, double f0x3F3C71C72D1B5154)
  %74 = tail call double @llvm.fma.f64(double %73, double %69, double f0x3F624924923BE72D)
  %75 = tail call double @llvm.fma.f64(double %74, double %69, double f0x3F8999999999A3C4)
  %76 = tail call double @llvm.fma.f64(double %75, double %69, double f0x3FB5555555555554)
  %77 = fsub double %60, %68
  %78 = fmul double %77, 2.000000e+00
  %79 = fneg double %68
  %80 = tail call double @llvm.fma.f64(double %79, double %60, double %78)
  %81 = fmul double %66, %80
  %82 = fmul double %69, %76
  %83 = tail call double @llvm.fma.f64(double %82, double %68, double %81)
  %84 = xor i32 %spec.select3, -2147483648
  %85 = tail call double @llvm.nvvm.lohi.i2d(i32 %84, i32 1127219200) #5
  %86 = tail call double @llvm.nvvm.lohi.i2d(i32 -2147483648, i32 1127219200) #5
  %87 = fsub double %85, %86
  %88 = tail call double @llvm.fma.f64(double %87, double f0x3FE62E42FEFA39EF, double %68)
  %89 = fneg double %87
  %90 = tail call double @llvm.fma.f64(double %89, double f0x3FE62E42FEFA39EF, double %88)
  %91 = fsub double %90, %68
  %92 = fsub double %83, %91
  %93 = tail call double @llvm.fma.f64(double %87, double f0x3C7ABC9E3B39803F, double %92)
  %94 = fadd double %88, %93
  br label %__nv_log1p.exit

95:                                               ; preds = %46
  %96 = tail call double @llvm.fma.f64(double %.0.i, double +inf, double +inf)
  %97 = tail call i32 @llvm.nvvm.d2i.hi(double %.0.i) #5
  %98 = and i32 %97, 2147483647
  %99 = icmp eq i32 %98, 0
  %q.i.0.i = select i1 %99, double -inf, double %96
  br label %__nv_log1p.exit

__nv_log1p.exit:                                  ; preds = %20, %48, %95
  %t.0.i = phi double [ %36, %20 ], [ %94, %48 ], [ %q.i.0.i, %95 ]
  %100 = fneg double %t.0.i
  %101 = fcmp ogt double %t.0.i, -6.250000e+00
  %102 = select i1 %101, double f0xBBB135D2E746E627, double f0x3E23040F87DBD932
  %103 = select i1 %101, double f0xBC08DDF93324D327, double f0x3E785CBE52878635
  %104 = fcmp ogt double %t.0.i, -1.600000e+01
  %105 = select i1 %104, double %102, double f0xBDBDCEC3A7785389
  %106 = select i1 %104, double %103, double f0xBDF18FEEC0E38727
  %107 = tail call double @llvm.sqrt.f64(double %100)
  %108 = select i1 %104, double 3.250000e+00, double 5.000000e+00
  %109 = fsub double -3.125000e+00, %t.0.i
  %110 = fsub double %107, %108
  %111 = select i1 %101, double %109, double %110
  %112 = fmul double %105, %111
  %113 = select i1 %101, double f0x3C37B83EEF0B7C9F, double f0xBE92777453DD3955
  %114 = fadd double %106, %112
  %115 = select i1 %104, double %113, double f0x3E19E6BF2DDA45E3
  %116 = fmul double %111, %114
  %117 = select i1 %101, double f0x3C69BA72CD589B91, double f0x3E5395ABCD554C6C
  %118 = fadd double %115, %116
  %119 = select i1 %104, double %117, double f0xBE30468FB24E2F5F
  %120 = fmul double %111, %118
  %121 = select i1 %101, double f0xBCA33689090A6B96, double f0x3EB936388A3790AD
  %122 = fadd double %119, %120
  %123 = select i1 %104, double %121, double f0x3E405AC6A8FBA182
  %124 = fmul double %111, %122
  %125 = select i1 %101, double f0x3C782E11898132E0, double f0xBED0D5DB812B5083
  %126 = fadd double %123, %124
  %127 = select i1 %104, double %125, double f0xBE50102E495FB9C0
  %128 = fmul double %111, %126
  %129 = select i1 %101, double f0x3CFDE4ACFD9E26BA, double f0x3EC8860CD5D652F6
  %130 = fadd double %127, %128
  %131 = select i1 %104, double %129, double f0x3E5F4C20E1334AF8
  %132 = fmul double %111, %130
  %133 = select i1 %101, double f0xBD26D33EED66C487, double f0x3EEA29A0CACDFB23
  %134 = fadd double %131, %132
  %135 = select i1 %104, double %133, double f0xBE722D220FDF9C3E
  %136 = fmul double %111, %134
  %137 = select i1 %101, double f0xBD36F2167040D8E2, double f0xBF08CEF1F80281F2
  %138 = fadd double %135, %136
  %139 = select i1 %104, double %137, double f0x3E8EBC8BB824CB54
  %140 = fmul double %111, %138
  %141 = select i1 %101, double f0x3D872A22C2D77E20, double f0x3F11E684D0B9188A
  %142 = fadd double %139, %140
  %143 = select i1 %104, double %141, double f0xBEB0A8D40EA372CC
  %144 = fmul double %111, %142
  %145 = select i1 %101, double f0xBDAC8859C4E5C0AF, double f0x3EF932CD54C8A222
  %146 = fadd double %143, %144
  %147 = select i1 %104, double %145, double f0x3ED2FBD29D093D2B
  %148 = fmul double %111, %146
  %149 = select i1 %101, double f0xBDCDC583D118A561, double f0xBF37448A89EF8AA3
  %150 = fadd double %147, %148
  %151 = select i1 %104, double %149, double f0xBEF4A3497E1E0FAC
  %152 = fmul double %111, %150
  %153 = select i1 %101, double f0x3E120F47CCF46B3C, double f0x3F4F3CC55AD40C25
  %154 = fadd double %151, %152
  %155 = select i1 %104, double %153, double f0x3F13EBF4EB00938F
  %156 = fmul double %111, %154
  %157 = select i1 %101, double f0xBE31A9E38DC84D60, double f0xBF5BA924132F38B1
  %158 = fadd double %155, %156
  %159 = select i1 %104, double %157, double f0xBF2C2F36A8FC5D53
  %160 = fmul double %111, %158
  %161 = select i1 %101, double f0xBE5F36CD6D3D46A9, double f0x3F6468EECA533CF8
  %162 = fadd double %159, %160
  %163 = select i1 %104, double %161, double f0xBF222EA5DF04047C
  %164 = fmul double %111, %162
  %165 = select i1 %101, double f0x3E9C6B4F5D03B787, double f0xBF6EBADABB891BBD
  %166 = fadd double %163, %164
  %167 = select i1 %104, double %165, double f0x3FF02A30D1FBA0DC
  %168 = fmul double %111, %166
  %169 = select i1 %101, double f0xBEB6E8A5434AE8A2, double f0x3F75FFCFE5B76AFC
  %170 = fadd double %167, %168
  %171 = select i1 %101, double f0xBEED1D1F7B8736F6, double f0x3FF0158A6D641D39
  %172 = select i1 %104, double %169, double f0x4013664DDD1AD7FB
  %173 = fmul double %111, %170
  %174 = fadd double %172, %173
  %175 = fmul double %111, %174
  %176 = select i1 %101, double f0x3F2879C2A212F024, double f0x4008ABCC380D5A48
  %177 = fadd double %171, %175
  %178 = select i1 %104, double %177, double %174
  %179 = fmul double %111, %178
  %180 = fadd double %176, %179
  %181 = select i1 %104, double %180, double %174
  %182 = fmul double %111, %181
  %183 = fadd double %182, f0xBF4845769484FCA8
  %184 = select i1 %101, double %183, double %181
  %185 = fmul double %111, %184
  %186 = fadd double %185, f0xBF78B6C33114F909
  %187 = select i1 %101, double %186, double %181
  %188 = fmul double %111, %187
  %189 = fadd double %188, f0x3FCEBD80D9B13E28
  %190 = select i1 %101, double %189, double %181
  %191 = fmul double %111, %190
  %192 = fadd double %191, f0x3FFA755E7C99AE86
  %193 = tail call double @llvm.nvvm.fabs.f64(double %13)
  %194 = select i1 %101, double %192, double %181
  %195 = fcmp oeq double %193, 1.000000e+00
  %.v = select i1 %195, double +inf, double %194
  %196 = fmul double %13, %.v
  %197 = fmul double %196, f0x3FF6A09E667F3BCD
  %198 = getelementptr inbounds [8 x i8], ptr addrspace(1) %5, i64 %11
  %199 = load double, ptr addrspace(1) %198, align 8, !invariant.load !6
  %200 = fmul double %197, 0.000000e+00
  %201 = fadd double %197, 0.000000e+00
  %202 = fadd double %199, %200
  %203 = fmul double %202, f0x3FE6A09E667F3BCC
  %204 = fmul double %201, 0.000000e+00
  %205 = fmul double %201, f0x3FE6A09E667F3BCC
  %206 = fmul double %202, 0.000000e+00
  %207 = fsub double %203, %204
  %208 = fadd double %205, %206
  %209 = getelementptr inbounds [16 x i8], ptr addrspace(1) %6, i64 %11
  %210 = insertelement <2 x double> poison, double %207, i32 0
  %211 = insertelement <2 x double> %210, double %208, i32 1
  store <2 x double> %211, ptr addrspace(1) %209, align 16
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.nvvm.fabs.f64(double) #2

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.nvvm.d2i.hi(double) #2

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.nvvm.d2i.lo(double) #2

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.nvvm.lohi.i2d(i32, i32) #2

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare double @llvm.nvvm.rcp.approx.ftz.d(double) #3

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.sqrt.f64(double) #4

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.fma.f64(double, double, double) #4

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) "nvvm.reqntid"="128,1,1" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none) }
attributes #4 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #5 = { nounwind }

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
