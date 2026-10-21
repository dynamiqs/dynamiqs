; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
define ptx_kernel void @loop_multiply_fusion(ptr noalias nofree readonly align 16 captures(none) dereferenceable(2048) %0, ptr noalias nofree readonly align 16 captures(none) dereferenceable(1024) %1, ptr noalias nofree writeonly align 256 captures(none) dereferenceable(3072) %2) local_unnamed_addr #0 {
  %4 = addrspacecast ptr %0 to ptr addrspace(1)
  %5 = addrspacecast ptr %1 to ptr addrspace(1)
  %6 = addrspacecast ptr %2 to ptr addrspace(1)
  %7 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !2
  %8 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !3
  %9 = shl nuw nsw i32 %7, 7
  %10 = or disjoint i32 %9, %8
  %11 = icmp samesign ult i32 %10, 192
  br i1 %11, label %12, label %140

12:                                               ; preds = %3
  %13 = shl nuw nsw i32 %7, 1
  %14 = lshr i32 %8, 6
  %15 = or disjoint i32 %13, %14
  %16 = urem i32 %15, 3
  %17 = and i32 %8, 63
  %18 = icmp samesign ugt i32 %16, 1
  br i1 %18, label %19, label %50

19:                                               ; preds = %12
  %20 = icmp samesign ugt i32 %17, 1
  br i1 %20, label %21, label %34

21:                                               ; preds = %19
  %22 = zext nneg i32 %17 to i64
  %23 = getelementptr [16 x i8], ptr addrspace(1) %4, i64 %22
  %24 = getelementptr i8, ptr addrspace(1) %23, i64 -32
  %25 = load <2 x double>, ptr addrspace(1) %24, align 16, !invariant.load !4, !alias.scope !5
  %.unpack.i41 = extractelement <2 x double> %25, i32 0
  %.unpack3.i42 = extractelement <2 x double> %25, i32 1
  %26 = or i32 %8, 64
  %27 = zext nneg i32 %26 to i64
  %28 = getelementptr inbounds [16 x i8], ptr addrspace(1) %4, i64 %27
  %29 = load <2 x double>, ptr addrspace(1) %28, align 16, !invariant.load !4
  %.unpack39 = extractelement <2 x double> %29, i32 0
  %.unpack540 = extractelement <2 x double> %29, i32 1
  %30 = fmul double %.unpack.i41, %.unpack39
  %31 = fmul double %.unpack3.i42, %.unpack540
  %32 = fadd double %30, %31
  %33 = fadd double %32, 0.000000e+00
  br label %34

34:                                               ; preds = %21, %19
  %35 = phi double [ %33, %21 ], [ 0.000000e+00, %19 ]
  %36 = icmp ugt i32 %17, 1
  br i1 %36, label %37, label %fused_multiply_dynamic_update_slice_222_1.exit

37:                                               ; preds = %34
  %38 = zext nneg i32 %17 to i64
  %39 = getelementptr [16 x i8], ptr addrspace(1) %4, i64 %38
  %40 = getelementptr i8, ptr addrspace(1) %39, i64 -32
  %41 = load <2 x double>, ptr addrspace(1) %40, align 16, !invariant.load !4, !alias.scope !8
  %.unpack.i1237 = extractelement <2 x double> %41, i32 0
  %.unpack3.i1338 = extractelement <2 x double> %41, i32 1
  %42 = or i32 %8, 64
  %43 = zext nneg i32 %42 to i64
  %44 = getelementptr inbounds [16 x i8], ptr addrspace(1) %4, i64 %43
  %45 = load <2 x double>, ptr addrspace(1) %44, align 16, !invariant.load !4
  %.unpack635 = extractelement <2 x double> %45, i32 0
  %.unpack836 = extractelement <2 x double> %45, i32 1
  %46 = fmul double %.unpack.i1237, %.unpack836
  %47 = fmul double %.unpack3.i1338, %.unpack635
  %48 = fsub double %46, %47
  %49 = fadd double %48, 0.000000e+00
  br label %fused_multiply_dynamic_update_slice_222_1.exit

50:                                               ; preds = %12
  tail call void @llvm.experimental.noalias.scope.decl(metadata !11)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !14)
  %51 = icmp eq i32 %16, 1
  %52 = zext i32 %17 to i64
  br i1 %51, label %53, label %77

53:                                               ; preds = %50
  %54 = icmp samesign ugt i32 %17, 1
  %55 = getelementptr [16 x i8], ptr addrspace(1) %4, i64 %52
  %56 = load <2 x double>, ptr addrspace(1) %55, align 16, !invariant.load !4, !alias.scope !16, !noalias !14
  %.unpack.i.i.i.i.i53 = extractelement <2 x double> %56, i32 0
  %.unpack3.i.i.i.i.i54 = extractelement <2 x double> %56, i32 1
  %57 = fmul double %.unpack.i.i.i.i.i53, %.unpack.i.i.i.i.i53
  %58 = fmul double %.unpack3.i.i.i.i.i54, %.unpack3.i.i.i.i.i54
  %59 = fadd double %57, %58
  br i1 %54, label %60, label %fused_multiply_dynamic_update_slice_218_3.exit.i

60:                                               ; preds = %53
  %61 = getelementptr i8, ptr addrspace(1) %55, i64 1024
  %62 = load <2 x double>, ptr addrspace(1) %61, align 16, !invariant.load !4, !alias.scope !19, !noalias !14
  %.unpack4.i.i.i.i51 = extractelement <2 x double> %62, i32 0
  %.unpack6.i.i.i.i52 = extractelement <2 x double> %62, i32 1
  %63 = fmul double %.unpack4.i.i.i.i51, %.unpack4.i.i.i.i51
  %64 = fmul double %.unpack6.i.i.i.i52, %.unpack6.i.i.i.i52
  %65 = fadd double %63, %64
  %66 = fadd double %59, %65
  br label %fused_multiply_dynamic_update_slice_218_3.exit.i

fused_multiply_dynamic_update_slice_218_3.exit.i: ; preds = %60, %53
  %67 = phi double [ %66, %60 ], [ %59, %53 ]
  %.not.i = icmp eq i32 %17, 0
  br i1 %.not.i, label %74, label %68

68:                                               ; preds = %fused_multiply_dynamic_update_slice_218_3.exit.i
  %69 = getelementptr [16 x i8], ptr addrspace(1) %5, i64 %52
  %70 = load <2 x double>, ptr addrspace(1) %69, align 16, !invariant.load !4, !alias.scope !24, !noalias !11
  %.unpack.i.i49 = extractelement <2 x double> %70, i32 0
  %.unpack2.i.i50 = extractelement <2 x double> %70, i32 1
  %71 = fmul double %.unpack.i.i49, %.unpack.i.i49
  %72 = fmul double %.unpack2.i.i50, %.unpack2.i.i50
  %73 = fadd double %71, %72
  br label %74

74:                                               ; preds = %68, %fused_multiply_dynamic_update_slice_218_3.exit.i
  %75 = phi double [ %73, %68 ], [ 0.000000e+00, %fused_multiply_dynamic_update_slice_218_3.exit.i ]
  %76 = fadd double %67, %75
  br label %fused_multiply_dynamic_update_slice_219_1.exit

77:                                               ; preds = %50
  %78 = icmp samesign ult i32 %17, 62
  %79 = getelementptr [16 x i8], ptr addrspace(1) %4, i64 %52
  br i1 %78, label %80, label %fused_multiply_mul_410_7.exit.i.i

80:                                               ; preds = %77
  %sunkaddr = getelementptr i8, ptr addrspace(1) %79, i64 1056
  %81 = load <2 x double>, ptr addrspace(1) %sunkaddr, align 16, !invariant.load !4, !alias.scope !27, !noalias !14
  %.unpack4.i.i10.i.i57 = extractelement <2 x double> %81, i32 0
  %.unpack6.i.i11.i.i58 = extractelement <2 x double> %81, i32 1
  %82 = insertvalue { double, double } poison, double %.unpack4.i.i10.i.i57, 0
  %83 = insertvalue { double, double } %82, double %.unpack6.i.i11.i.i58, 1
  br label %fused_multiply_mul_410_7.exit.i.i

fused_multiply_mul_410_7.exit.i.i:                ; preds = %80, %77
  %84 = phi { double, double } [ %83, %80 ], [ zeroinitializer, %77 ]
  %85 = extractvalue { double, double } %84, 1
  %86 = load <2 x double>, ptr addrspace(1) %79, align 16, !invariant.load !4, !alias.scope !34, !noalias !14
  %.unpack.i.i.i55 = extractelement <2 x double> %86, i32 0
  %.unpack2.i.i.i56 = extractelement <2 x double> %86, i32 1
  %87 = extractvalue { double, double } %84, 0
  %88 = fmul double %87, %.unpack.i.i.i55
  %89 = fmul double %85, %.unpack2.i.i.i56
  %90 = fadd double %88, %89
  br label %fused_multiply_dynamic_update_slice_219_1.exit

fused_multiply_dynamic_update_slice_219_1.exit:   ; preds = %74, %fused_multiply_mul_410_7.exit.i.i
  %.unpack3.i.i.i.i.i23 = phi double [ %.unpack3.i.i.i.i.i54, %74 ], [ %.unpack2.i.i.i56, %fused_multiply_mul_410_7.exit.i.i ]
  %.unpack.i.i.i.i.i22 = phi double [ %.unpack.i.i.i.i.i53, %74 ], [ %.unpack.i.i.i55, %fused_multiply_mul_410_7.exit.i.i ]
  %91 = phi double [ %76, %74 ], [ %90, %fused_multiply_mul_410_7.exit.i.i ]
  %92 = icmp eq i32 %16, 1
  tail call void @llvm.experimental.noalias.scope.decl(metadata !35)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !38)
  br i1 %92, label %93, label %113

93:                                               ; preds = %fused_multiply_dynamic_update_slice_219_1.exit
  %94 = icmp samesign ugt i32 %17, 1
  %95 = fmul double %.unpack.i.i.i.i.i22, %.unpack3.i.i.i.i.i23
  %96 = fsub double %95, %95
  br i1 %94, label %97, label %fused_multiply_dynamic_update_slice_221_3.exit.i

97:                                               ; preds = %93
  %98 = getelementptr [16 x i8], ptr addrspace(1) %4, i64 %52
  %99 = getelementptr i8, ptr addrspace(1) %98, i64 1024
  %100 = load <2 x double>, ptr addrspace(1) %99, align 16, !invariant.load !4, !alias.scope !40, !noalias !38
  %.unpack4.i.i.i.i2845 = extractelement <2 x double> %100, i32 0
  %.unpack6.i.i.i.i3046 = extractelement <2 x double> %100, i32 1
  %101 = fmul double %.unpack4.i.i.i.i2845, %.unpack6.i.i.i.i3046
  %102 = fsub double %101, %101
  %103 = fadd double %96, %102
  br label %fused_multiply_dynamic_update_slice_221_3.exit.i

fused_multiply_dynamic_update_slice_221_3.exit.i: ; preds = %97, %93
  %104 = phi double [ %103, %97 ], [ %96, %93 ]
  %.not.i24 = icmp eq i32 %17, 0
  %105 = getelementptr [16 x i8], ptr addrspace(1) %5, i64 %52
  br i1 %.not.i24, label %110, label %106

106:                                              ; preds = %fused_multiply_dynamic_update_slice_221_3.exit.i
  %107 = load <2 x double>, ptr addrspace(1) %105, align 16, !invariant.load !4, !alias.scope !47, !noalias !35
  %.unpack.i.i2543 = extractelement <2 x double> %107, i32 0
  %.unpack2.i.i2744 = extractelement <2 x double> %107, i32 1
  %108 = fmul double %.unpack.i.i2543, %.unpack2.i.i2744
  %109 = fsub double %108, %108
  br label %110

110:                                              ; preds = %106, %fused_multiply_dynamic_update_slice_221_3.exit.i
  %111 = phi double [ %109, %106 ], [ 0.000000e+00, %fused_multiply_dynamic_update_slice_221_3.exit.i ]
  %112 = fadd double %104, %111
  br label %fused_multiply_dynamic_update_slice_222_1.exit

113:                                              ; preds = %fused_multiply_dynamic_update_slice_219_1.exit
  %114 = icmp samesign ult i32 %17, 62
  %115 = getelementptr [16 x i8], ptr addrspace(1) %4, i64 %52
  br i1 %114, label %116, label %fused_multiply_mul_410_7.exit.i.i14

116:                                              ; preds = %113
  %sunkaddr59 = getelementptr i8, ptr addrspace(1) %115, i64 1056
  %117 = load <2 x double>, ptr addrspace(1) %sunkaddr59, align 16, !invariant.load !4, !alias.scope !50, !noalias !38
  %.unpack4.i.i10.i.i1947 = extractelement <2 x double> %117, i32 0
  %.unpack6.i.i11.i.i2048 = extractelement <2 x double> %117, i32 1
  %118 = insertvalue { double, double } poison, double %.unpack4.i.i10.i.i1947, 0
  %119 = insertvalue { double, double } %118, double %.unpack6.i.i11.i.i2048, 1
  br label %fused_multiply_mul_410_7.exit.i.i14

fused_multiply_mul_410_7.exit.i.i14:              ; preds = %116, %113
  %120 = phi { double, double } [ %119, %116 ], [ zeroinitializer, %113 ]
  %121 = extractvalue { double, double } %120, 1
  %122 = extractvalue { double, double } %120, 0
  %123 = fmul double %122, %.unpack3.i.i.i.i.i23
  %124 = fmul double %121, %.unpack.i.i.i.i.i22
  %125 = fsub double %123, %124
  br label %fused_multiply_dynamic_update_slice_222_1.exit

fused_multiply_dynamic_update_slice_222_1.exit:   ; preds = %34, %37, %fused_multiply_mul_410_7.exit.i.i14, %110
  %126 = phi double [ %91, %fused_multiply_mul_410_7.exit.i.i14 ], [ %91, %110 ], [ %35, %37 ], [ %35, %34 ]
  %127 = phi double [ %125, %fused_multiply_mul_410_7.exit.i.i14 ], [ %112, %110 ], [ %49, %37 ], [ 0.000000e+00, %34 ]
  %128 = fmul double %126, -5.000000e-01
  %129 = fmul double %127, 0.000000e+00
  %130 = fmul double %127, 5.000000e-01
  %131 = fmul double %126, 0.000000e+00
  %132 = fsub double %128, %129
  %133 = fsub double %131, %130
  %134 = shl nuw nsw i32 %16, 6
  %135 = or disjoint i32 %134, %17
  %136 = zext nneg i32 %135 to i64
  %137 = getelementptr inbounds [16 x i8], ptr addrspace(1) %6, i64 %136
  %138 = insertelement <2 x double> poison, double %132, i32 0
  %139 = insertelement <2 x double> %138, double %133, i32 1
  store <2 x double> %139, ptr addrspace(1) %137, align 16
  br label %140

140:                                              ; preds = %fused_multiply_dynamic_update_slice_222_1.exit, %3
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare void @llvm.experimental.noalias.scope.decl(metadata) #2

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite) "nvvm.reqntid"="128,1,1" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }

!llvm.module.flags = !{!0, !1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!2 = !{i32 0, i32 2}
!3 = !{i32 0, i32 128}
!4 = !{}
!5 = !{!6}
!6 = distinct !{!6, !7, !"fused_multiply_complex_32_28: argument 0"}
!7 = distinct !{!7, !"fused_multiply_complex_32_28"}
!8 = !{!9}
!9 = distinct !{!9, !10, !"fused_multiply_complex_32_28: argument 0"}
!10 = distinct !{!10, !"fused_multiply_complex_32_28"}
!11 = !{!12}
!12 = distinct !{!12, !13, !"fused_multiply_dynamic_update_slice_219_1: argument 0"}
!13 = distinct !{!13, !"fused_multiply_dynamic_update_slice_219_1"}
!14 = !{!15}
!15 = distinct !{!15, !13, !"fused_multiply_dynamic_update_slice_219_1: argument 1"}
!16 = !{!17, !12}
!17 = distinct !{!17, !18, !"fused_multiply_dynamic_update_slice_218_3: argument 0"}
!18 = distinct !{!18, !"fused_multiply_dynamic_update_slice_218_3"}
!19 = !{!20, !22, !17, !12}
!20 = distinct !{!20, !21, !"fused_multiply_complex_32_28: argument 0"}
!21 = distinct !{!21, !"fused_multiply_complex_32_28"}
!22 = distinct !{!22, !23, !"fused_multiply_mul_412_3: argument 0"}
!23 = distinct !{!23, !"fused_multiply_mul_412_3"}
!24 = !{!25, !15}
!25 = distinct !{!25, !26, !"fused_multiply_mul_414_3: argument 0"}
!26 = distinct !{!26, !"fused_multiply_mul_414_3"}
!27 = !{!28, !30, !32, !12}
!28 = distinct !{!28, !29, !"fused_multiply_complex_32_28: argument 0"}
!29 = distinct !{!29, !"fused_multiply_complex_32_28"}
!30 = distinct !{!30, !31, !"fused_multiply_mul_410_7: argument 0"}
!31 = distinct !{!31, !"fused_multiply_mul_410_7"}
!32 = distinct !{!32, !33, !"fused_multiply_dynamic_update_slice_218_3: argument 0"}
!33 = distinct !{!33, !"fused_multiply_dynamic_update_slice_218_3"}
!34 = !{!30, !32, !12}
!35 = !{!36}
!36 = distinct !{!36, !37, !"fused_multiply_dynamic_update_slice_222_1: argument 0"}
!37 = distinct !{!37, !"fused_multiply_dynamic_update_slice_222_1"}
!38 = !{!39}
!39 = distinct !{!39, !37, !"fused_multiply_dynamic_update_slice_222_1: argument 1"}
!40 = !{!41, !43, !45, !36}
!41 = distinct !{!41, !42, !"fused_multiply_complex_32_28: argument 0"}
!42 = distinct !{!42, !"fused_multiply_complex_32_28"}
!43 = distinct !{!43, !44, !"fused_multiply_mul_412_3: argument 0"}
!44 = distinct !{!44, !"fused_multiply_mul_412_3"}
!45 = distinct !{!45, !46, !"fused_multiply_dynamic_update_slice_221_3: argument 0"}
!46 = distinct !{!46, !"fused_multiply_dynamic_update_slice_221_3"}
!47 = !{!48, !39}
!48 = distinct !{!48, !49, !"fused_multiply_mul_414_3: argument 0"}
!49 = distinct !{!49, !"fused_multiply_mul_414_3"}
!50 = !{!51, !53, !55, !36}
!51 = distinct !{!51, !52, !"fused_multiply_complex_32_28: argument 0"}
!52 = distinct !{!52, !"fused_multiply_complex_32_28"}
!53 = distinct !{!53, !54, !"fused_multiply_mul_410_7: argument 0"}
!54 = distinct !{!54, !"fused_multiply_mul_410_7"}
!55 = distinct !{!55, !56, !"fused_multiply_dynamic_update_slice_221_3: argument 0"}
!56 = distinct !{!56, !"fused_multiply_dynamic_update_slice_221_3"}
