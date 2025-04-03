$(document).ready(function() {
    $(document).on("click", "a.modal-form", function (e) {
        e.preventDefault();
    });
    $(".modal-form").each(function () {
        $(this).modalForm({
            formURL: $(this).attr("href")
        });
    });
    $(".modal").on("show.bs.modal", function (event) {
        formset.init();

        $(".rang-input").each(function () {
            $(this).closest(".form-group").find(".rang-value").html(this.value);
        }).on("input", function () {
            $(this).closest(".form-group").find(".rang-value").html(this.value);
        });
    });
});