"""Render the unsmoothed training and validation histories of the complete matrix."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

from .analyze import read_json, completed_units
from .model import ENCODERS, DECODERS
from .run import ROOT, write_json, file_records
from .data import file_hash


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=ROOT/'results/N012_independent')
    args=parser.parse_args()
    output=args.out.resolve()
    destination=ROOT/'report/N012/figures'
    destination.mkdir(parents=True,exist_ok=True)
    font_path=Path('C:/Windows/Fonts/msyh.ttc')
    font=FontProperties(fname=str(font_path)) if font_path.exists() else FontProperties()
    histories=[]
    figures=[]
    colors=('#0072B2','#D55E00','#009E73','#CC79A7','#333333')
    for domain in ('MATR','HUST'):
        registration,units=completed_units(output,domain)
        n1=read_json(output/domain/'N1_SELECTION.json')
        n2=read_json(output/domain/'N2_SELECTION.json')
        folds=n1['selection_folds']
        for phase in ('N1','N2'):
            figure,axes=plt.subplots(5,2,figsize=(12,15),sharex=True)
            candidates=ENCODERS if phase=='N1' else DECODERS
            for row,fold in enumerate(folds):
                selected=n1['encoder_per_outer_fold'][fold] if phase=='N1' else n2['selected_decoder_per_fold'][fold]
                encoder=n1['encoder_per_outer_fold'][fold]
                for color,candidate in zip(colors,candidates):
                    key=(fold,candidate,'query_mlp') if phase=='N1' else (fold,encoder,candidate)
                    path=units[key]['root']/'HISTORY.json'
                    history=read_json(path)
                    histories.append(dict(domain=domain,phase=phase,fold=fold,candidate=candidate,
                                          path=path.relative_to(ROOT).as_posix(),sha256=file_hash(path)))
                    epochs=[record['epoch'] for record in history]
                    axes[row,0].plot(epochs,[record['train_mae'] for record in history],color=color,linewidth=1.1,label=candidate)
                    axes[row,1].plot(epochs,[record['val_mae'] for record in history],color=color,linewidth=1.1,label=candidate)
                    if candidate==selected:
                        best=min(history,key=lambda record:record['val_mae'])
                        axes[row,1].scatter([best['epoch']],[best['val_mae']],marker='o',s=25,color=color,zorder=5)
                label=fold if phase=='N1' else fold+' / '+encoder
                for column in (0,1):
                    axes[row,column].set_title(label,fontsize=9,loc='left')
                    axes[row,column].set_ylim(bottom=0)
                    axes[row,column].grid(alpha=.18)
                    axes[row,column].set_ylabel('逐电池 MAE',fontproperties=font,fontsize=9)
                    axes[row,column].tick_params(labelsize=8)
            axes[0,0].set_title('训练集',fontproperties=font,fontsize=12,loc='right')
            axes[0,1].set_title('验证集',fontproperties=font,fontsize=12,loc='right')
            for axis in axes[-1]:
                axis.set_xlabel('训练轮数',fontproperties=font)
            handles,labels=axes[0,0].get_legend_handles_labels()
            figure.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.973),ncol=3,frameon=False,fontsize=10)
            figure.suptitle(domain+' '+phase+' 原始训练与验证曲线',fontproperties=font,fontsize=16,y=.995)
            figure.tight_layout(rect=(0,0,1,.94),h_pad=1.5,w_pad=2)
            path=destination/(domain+'_'+phase+'_training_validation.png')
            figure.savefig(path,dpi=180,facecolor='white')
            plt.close(figure)
            figures.append(path.name)
    write_json(destination/'FIGURES_COMPLETE.json',dict(status='RENDERED',source_sha256=file_hash(__file__),
               histories=histories,files=file_records(destination,figures),smoothing=False,optimizer_updates=0))
    print('Rendered4 complete matrix history figures',flush=True)


if __name__=='__main__':
    main()
