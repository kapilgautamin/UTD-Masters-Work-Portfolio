node --version
npm install -g express-generator
express --version
cd desktop
express Vidzy

cd Vidzy
npm install 
npm install nodemon -g
nodemon

npm install monk --save 
sudo npm install ejs --save
npm install method-override --save 


mongod                      // to start mongodb as a service
mongod --dbpath ~/data/db   // if your dbpath is different from /data/db

Steps for displaying movies on start/welcome page

1. index.js: retrieve all documents from mongodb, pass them to ejs file
2. index.ejs: display all videos
3. create partials folder under views folder
4. create header.ejs and footer.ejs under partials folder
5. index.ejs: include header and footer ejs files
6. include bootstrap in header.ejs
7. Download the images and save them under public/images folder
8. Add a new attribute (images) to movie documents in mongodb
9. Show video list with images
10. Add jumbotron


Next steps
- Video insert (/videos/new)
- Show video details (/videos/:id)
- Delete a video