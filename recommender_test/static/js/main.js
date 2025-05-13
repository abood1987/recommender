$(document).ready(function() {
    $(document).on("click", "a.modal-form", function (e) {
        e.preventDefault();
    });
    $(".modal-form").each(function () {
        $(this).modalForm({
            formURL: $(this).attr("href"),
            submitBtn: ".modal-submit-btn"
        });
    });

  // // Optional: hide spinner when modal is closed
  //   $('#modal').on('hidden.bs.modal', function () {
  //       $('#form-spinner').hide();
  //   });

    $(".modal")
        .on("show.bs.modal", function (event) {
            formset.init();

            $(".rang-input").each(function () {
                $(this).closest(".form-group").find(".rang-value").html(this.value);
            }).on("input", function () {
                $(this).closest(".form-group").find(".rang-value").html(this.value);
            });
        })
        .on("submit", ".modal-content form", function () {
            $("#form-spinner").show();
        });
});