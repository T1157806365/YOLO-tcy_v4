cd /mnt/sda/taochangyong/Projects/Model/YOLO-tcy_v4 
 
# 先确认当前状态 
git status 
git branch --show-current 
git remote -v 
 
# 基于当前代码创建新的备份分支 
git switch -c backup-20261026_FigConvAdd
 
# 提交当前所有未被 .gitignore 排除的修改 
git add . 
 
git commit -m "Backup YOLO-tcy_v4 20261026_FigConvAdd" 
 
# 上传新分支到 GitHub 
git push -u origin backup-20261026_FigConvAdd